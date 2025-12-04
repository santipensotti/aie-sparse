#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using i32 = int32_t;
using i64 = int64_t;

enum class MMField { REAL, INTEGER, PATTERN, COMPLEX };
enum class MMSymm  { GENERAL, SYMMETRIC, SKEW, HERMITIAN };

struct MTXHeader {
  bool coordinate = false;
  MMField field = MMField::REAL;
  MMSymm  symm  = MMSymm::GENERAL;
  i64 M=0,N=0,NNZ=0;
  std::streampos data_pos{};
};
struct FlatLayout {
  i64 stride_bytes;
  std::vector<uint8_t> buf; // size = tiles.size() * stride_bytes
};


static std::string lower(std::string s){ for(char& c:s) c=char(::tolower(c)); return s; }

static MTXHeader read_mtx_header(std::ifstream& in) {
  MTXHeader h;
  std::string line;
  bool banner=false;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line[0]=='%') {
      if (!banner && line.rfind("%%MatrixMarket",0)==0) {
        std::istringstream is(line);
        std::string pct,obj,fmt,fld,sym; is>>pct>>obj>>fmt>>fld>>sym;
        if (lower(fmt)!="coordinate") throw std::runtime_error("Only 'coordinate' MTX supported");
        h.coordinate = true;
        std::string f=lower(fld), s=lower(sym);
        if      (f=="real")    h.field=MMField::REAL;
        else if (f=="integer") h.field=MMField::INTEGER;
        else if (f=="pattern") h.field=MMField::PATTERN;
        else if (f=="complex") h.field=MMField::COMPLEX;
        else throw std::runtime_error("Unknown field: "+fld);
        if      (s=="general")        h.symm=MMSymm::GENERAL;
        else if (s=="symmetric")      h.symm=MMSymm::SYMMETRIC;
        else if (s=="skew-symmetric") h.symm=MMSymm::SKEW;
        else if (s=="hermitian")      h.symm=MMSymm::HERMITIAN;
        else throw std::runtime_error("Unknown symmetry: "+sym);
        banner=true;
      }
      continue;
    } else {
      std::istringstream ls(line);
      if (!(ls>>h.M>>h.N>>h.NNZ)) throw std::runtime_error("Bad MTX size line");
      h.data_pos = in.tellg();
      break;
    }
  }
  if (!banner) throw std::runtime_error("Missing MatrixMarket banner");
  return h;
}

struct Tile {
  // rangos globales (para mapear a x_k)
  i64 row_start=0, row_end=0; // [)
  i64 col_start=0, col_end=0; // [)
  // buffers fijos
  i32 nrows_fixed=0;    // == row_block
  i32 ncols_fixed=0;    // == x_seg (última stripe puede usar menos; guardamos col_end-col_start)
  i64 cap_nnz=0;        // igual para todos
  std::vector<i64> rowptr;     // len = nrows_fixed+1 (últimas filas vacías si row_end-row_start < nrows_fixed)
  std::vector<i32> colidx;     // len = cap_nnz
  std::vector<double> vals;    // len = cap_nnz
  i64 nnz_used=0;       // cuántos válidos
  i32 nrows_used=0;     // row_end-row_start
  i32 ncols_used=0;     // col_end-col_start
};
FlatLayout pack_flat(const std::vector<Tile>& tiles, i64 row_block, i64 CAP) {
  // Layout por tile: [rowptr ( (row_block+1)*int64 ) | colidx (CAP*int32) | vals (CAP*double) ]
  i64 bytes_rowptr = (row_block + 1) * sizeof(int64_t);
  i64 bytes_colidx = CAP * sizeof(int32_t);
  i64 bytes_vals   = CAP * sizeof(double);
  i64 stride = bytes_rowptr + bytes_colidx + bytes_vals;

  FlatLayout out;
  out.stride_bytes = stride;
  out.buf.resize(size_t(tiles.size()) * size_t(stride));

  for (size_t t=0; t<tiles.size(); ++t) {
    const auto& T = tiles[t];
    uint8_t* base = out.buf.data() + t*stride;
    // copiar rowptr
    std::memcpy(base, T.rowptr.data(), bytes_rowptr);
    // copiar colidx
    std::memcpy(base + bytes_rowptr, T.colidx.data(), bytes_colidx);
    // copiar vals
    std::memcpy(base + bytes_rowptr + bytes_colidx, T.vals.data(), bytes_vals);
  }
  return out;
}

struct Args {
  std::string mtx;
  i64 x_seg=4096;
  i64 row_block=4096;
  i64 align=64;
  bool no_sym_expand=false;
};

static void usage(){
  std::cerr<<"Usage: two_step_fixed --mtx A.mtx --x-seg W --row-block R [--align A] [--no-symmetric-expand]\n";
}
static Args parse_args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];

    if (s == "--help" || s == "-h") {
      usage();
      std::exit(0);
    } else if (s == "--no-symmetric-expand") {
      a.no_sym_expand = true;
    } else if (s.rfind("--mtx=", 0) == 0) {
      a.mtx = s.substr(6);
    } else if (s.rfind("--x-seg=", 0) == 0) {
      a.x_seg = std::stoll(s.substr(8));
    } else if (s.rfind("--row-block=", 0) == 0) {
      a.row_block = std::stoll(s.substr(12));
    } else if (s.rfind("--align=", 0) == 0) {
      a.align = std::stoll(s.substr(8));
    } else if (s == "--mtx" && i + 1 < argc) {
      a.mtx = argv[++i];
    } else if (s == "--x-seg" && i + 1 < argc) {
      a.x_seg = std::stoll(argv[++i]);
    } else if (s == "--row-block" && i + 1 < argc) {
      a.row_block = std::stoll(argv[++i]);
    } else if (s == "--align" && i + 1 < argc) {
      a.align = std::stoll(argv[++i]);
    } else {
      std::cerr << "Unknown arg: " << s << "\n";
      usage();
      std::exit(1);
    }
  }
  if (a.mtx.empty() || a.x_seg <= 0 || a.row_block <= 0) {
    usage();
    std::exit(1);
  }
  return a;
}

struct Counters {
  // por tile: nnz total
  std::vector<i64> tile_nnz;
  // por tile × fila_local: nnz por fila (para CSR)
  std::vector<i64> row_counts; // flattened
};

static inline i64 ceil_up(i64 x, i64 a){ return ((x + a - 1)/a)*a; }

// map (i,j) -> (k,r, i_local, j_local)
struct Mapper {
  i64 M,N,W,R,num_stripes,num_rowblocks;
  Mapper(i64 M_,i64 N_,i64 W_,i64 R_):M(M_),N(N_),W(W_),R(R_){
    num_stripes = (N + W - 1)/W;
    num_rowblocks = (M + R - 1)/R;
  }
  inline i64 stripe(i64 j) const { return j / W; }
  inline i64 rowblk(i64 i) const { return i / R; }
  inline i64 tile_index(i64 k,i64 r) const { return k*num_rowblocks + r; }
  inline i64 row_start(i64 r) const { return r*R; }
  inline i64 row_end(i64 r) const { return std::min((r+1)*R, M); }
  inline i64 col_start(i64 k) const { return k*W; }
  inline i64 col_end(i64 k) const { return std::min((k+1)*W, N); }
  inline i32 i_local(i64 i,i64 r) const { return (i32)(i - row_start(r)); }
  inline i32 j_local(i64 j,i64 k) const { return (i32)(j - col_start(k)); }
};

// ---------- Pass 1: contar ----------
static void pass1_count(const Args& a, const MTXHeader& h, Counters& C, const Mapper& map){
  std::ifstream in(a.mtx);
  in.seekg(h.data_pos);
  const i64 tiles = map.num_stripes * map.num_rowblocks;
  C.tile_nnz.assign(tiles, 0);
  C.row_counts.assign(tiles * a.row_block, 0); // último bloque puede tener menos filas; ok

  std::string ln;
  i64 processed=0;
  auto emit=[&](i64 i,i64 j){
    i64 k = map.stripe(j);
    i64 r = map.rowblk(i);
    i32 il = map.i_local(i,r);
    i64 tidx = map.tile_index(k,r);
    C.tile_nnz[tidx] += 1;
    C.row_counts[tidx*a.row_block + il] += 1;
  };

  while (std::getline(in, ln)) {
    if (ln.empty() || ln[0]=='%') continue;
    std::istringstream ss(ln);
    i64 ii,jj; if(!(ss>>ii>>jj)) continue;
    double rval=1.0, imag=0.0;
    if (h.field==MMField::REAL || h.field==MMField::INTEGER) { if(!(ss>>rval)) rval=0.0; }
    else if (h.field==MMField::COMPLEX) { if(!(ss>>rval>>imag)) {rval=0.0; imag=0.0;} }
    // 1-based -> 0-based
    i64 i=ii-1, j=jj-1;
    // general
    if (h.symm==MMSymm::GENERAL) emit(i,j);
    else if (h.symm==MMSymm::SYMMETRIC) {
      emit(i,j);
      if (!a.no_sym_expand && i!=j) emit(j,i);
    } else if (h.symm==MMSymm::SKEW) {
      emit(i,j);
      if (!a.no_sym_expand && i!=j) emit(j,i);
    } else if (h.symm==MMSymm::HERMITIAN) {
      emit(i,j);
      if (!a.no_sym_expand && i!=j) emit(j,i);
    }
    if(++processed && (processed%1000000==0)) {
      std::cerr<<"[pass1] "<<processed<<" entries\n";
    }
  }
}

// ---------- Reservar tiles con CAP uniforme ----------
static void allocate_tiles_uniform(std::vector<Tile>& tiles, const Args& a, const MTXHeader& h, const Mapper& map, const Counters& C){
  const i64 tilesN = map.num_stripes * map.num_rowblocks;
  tiles.resize(tilesN);
  // max nnz tile
  i64 max_nnz=0;
  for (i64 t=0;t<tilesN;++t) max_nnz = std::max(max_nnz, C.tile_nnz[t]);
  i64 CAP = std::max<i64>(1, ceil_up(max_nnz, a.align));

  for (i64 k=0;k<map.num_stripes;++k){
    for (i64 r=0;r<map.num_rowblocks;++r){
      i64 t = map.tile_index(k,r);
      auto& T = tiles[t];
      T.row_start = map.row_start(r);
      T.row_end   = map.row_end(r);
      T.col_start = map.col_start(k);
      T.col_end   = map.col_end(k);
      T.nrows_fixed = (i32)a.row_block;
      T.ncols_fixed = (i32)a.x_seg;
      T.nrows_used  = (i32)(T.row_end - T.row_start);
      T.ncols_used  = (i32)(T.col_end - T.col_start);
      T.cap_nnz = CAP;
      T.rowptr.assign(a.row_block + 1, 0); // fija
      T.colidx.assign(CAP, 0);
      T.vals.assign(CAP, 0.0);
      T.nnz_used = C.tile_nnz[t];
    }
  }

  // construir rowptr por tile desde row_counts
  for (i64 k=0;k<map.num_stripes;++k){
    for (i64 r=0;r<map.num_rowblocks;++r){
      i64 t = map.tile_index(k,r);
      auto& T = tiles[t];
      i64 base = t * a.row_block;
      i64 run=0;
      for (i64 il=0; il<a.row_block; ++il){
        T.rowptr[il] = run;
        run += C.row_counts[base + il];
      }
      T.rowptr[a.row_block] = run; // puede ser < CAP
    }
  }
}

// ---------- Pass 2: llenar ----------
static void pass2_fill(const Args& a, const MTXHeader& h, std::vector<Tile>& tiles, const Mapper& map){
  std::ifstream in(a.mtx);
  in.seekg(h.data_pos);
  // cursores por fila local (tile × R)
  std::vector<i64> cursor(tiles.size()*a.row_block, 0);
  for (size_t t=0;t<tiles.size();++t){
    auto& T = tiles[t];
    const i64 base = t*a.row_block;
    for (i64 il=0; il<a.row_block; ++il) cursor[base+il] = T.rowptr[il];
  }

  std::string ln; i64 processed=0;
  auto emit=[&](i64 i,i64 j,double v){
    i64 k = map.stripe(j);
    i64 r = map.rowblk(i);
    i64 t = map.tile_index(k,r);
    auto& T = tiles[t];
    i32 il = map.i_local(i,r);
    i32 jl = map.j_local(j,k);
    i64 pos = cursor[t*a.row_block + il]++;
    if (pos >= T.cap_nnz) throw std::runtime_error("Tile overflow: increase --align or change tiling");
    T.colidx[pos] = jl;
    T.vals[pos]   = v;
  };

  while (std::getline(in, ln)) {
    if (ln.empty() || ln[0]=='%') continue;
    std::istringstream ss(ln);
    i64 ii,jj; if(!(ss>>ii>>jj)) continue;
    double val=1.0, imag=0.0;
    if (h.field==MMField::REAL || h.field==MMField::INTEGER) { if(!(ss>>val)) val=0.0; }
    else if (h.field==MMField::COMPLEX) { if(!(ss>>val>>imag)) {val=0.0; imag=0.0;} }
    i64 i=ii-1, j=jj-1;

    if (h.symm==MMSymm::GENERAL) emit(i,j,val);
    else if (h.symm==MMSymm::SYMMETRIC) {
      emit(i,j,val);
      if (!a.no_sym_expand && i!=j) emit(j,i,val);
    } else if (h.symm==MMSymm::SKEW) {
      emit(i,j,val);
      if (!a.no_sym_expand && i!=j) emit(j,i,-val);
    } else if (h.symm==MMSymm::HERMITIAN) {
      emit(i,j,val);
      if (!a.no_sym_expand && i!=j) emit(j,i,val); // sin conj por simplicidad
    }

    if(++processed && (processed%1000000==0))
      std::cerr<<"[pass2] "<<processed<<" entries\n";
  }
}

int main(int argc,char**argv){
  try{
    Args a = parse_args(argc, argv);
    std::ifstream in(a.mtx);
    if(!in) { std::cerr<<"Cannot open "<<a.mtx<<"\n"; return 1; }
    MTXHeader h = read_mtx_header(in);
    std::cerr<<"[info] "<<h.M<<"x"<<h.N<<"  nnz="<<h.NNZ<<"\n";
    Mapper map(h.M, h.N, a.x_seg, a.row_block);
    std::cerr<<"[info] stripes="<<map.num_stripes<<" row_blocks="<<map.num_rowblocks<<"\n";

    // Pass 1: contar
    Counters C;
    pass1_count(a, h, C, map);

    // Reservar tiles con CAP uniforme (mismo tamaño para todos)
    std::vector<Tile> tiles;
    allocate_tiles_uniform(tiles, a, h, map, C);

    // Pass 2: llenar
    pass2_fill(a, h, tiles, map);

    // Resumen
    i64 max_used=0, min_used=LLONG_MAX, sum_used=0;
    for (auto& T: tiles) {
      max_used = std::max(max_used, T.nnz_used);
      min_used = std::min(min_used, T.nnz_used);
      sum_used += T.nnz_used;
    }
    std::cout<<"[summary] tiles="<<tiles.size()
             <<" cap="<<tiles[0].cap_nnz
             <<" used_avg="<<(double)sum_used/tiles.size()
             <<" used_min="<<min_used
             <<" used_max="<<max_used
             <<"\n";


    FlatLayout layout = pack_flat(tiles, a.row_block, tiles[0].cap_nnz);
    std::cout <<"[info] flat layout: stride_bytes="<<layout.stride_bytes
             <<" total_bytes="<<layout.buf.size()
             <<" tiles="<<layout.buf.size()/layout.stride_bytes
             <<"\n";
    return 0;
  } catch (const std::exception& e){
    std::cerr<<"Error: "<<e.what()<<"\n";
    return 1;
  }
}


// Con esto paso file -> buffer de tiles con CAP uniforme
// Cada tile tiene rowptr fijo (R+1), colidx (CAP), vals (CAP)
// CAP es el mismo para todos los tiles, alineado a 'align'
// Uso tiling por filas (row_block) y por columnas (x_seg)
// Necesito pensar como pasar a npu teniendo el problema de que me genera varias copias de Y.
// Tengo un problema que es que tengo que decir que tanta memoria reservar en el device. Pero a veces
// no uso todo el tamaño y si tambien hago varias pasadas por columna de X. para row, tengo que pensar como hacerlo.
// Puedo tratar de hacer que todas tengan la mmisma cantidada de nnz, pero eso no resuelve el problema de pasar menos Y.
// Si hago por bloques entonces se que cantidad de Y necesito por bloque. Pero como uno despues? Que pasa si un bloque no tiene elementos deY?

// Haciendo ping pong a memoria que soluciono? Si tengo cosas que no entran en memoria, puedo hacer ping pong a memoria. para que?
// Cada bloque va a tener su pedazo de Y. Despues tengo que juntar todo Y.
// Como hago eso?
// Tal vez puedo hacer que cada bloque escriba su pedazo de Y en memoria y despues hago un kernel que junte todo Y. Pero hay estoy escribiendo muchas veces a memoria.
// Aprovechando que ya tengo Y en memoria parcial, si uso fifo entre vecinos puedo ir sumando los pedazos de Y. Y achico la cantidad de memoria que necesito.
// En vez de 9 copias de Y, necesito 1. El costo es si un core termina rapido se queda esperando a los demas, pero si hago una distribucion pareja entonces puedo minimizar ese efecto.
// El flow del programa seria:
// 1. Cargo A en tiles en memoria del host.
// 2. Cargo X en memoria del host.
// 3. Paso tiles y X al device.
// 4. Cada core procesa su tile y va pasando Y por fifo a su vecino.
// 5. El ultimo core escribe Y en memoria global.
