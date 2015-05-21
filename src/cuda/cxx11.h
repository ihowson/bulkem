// There are a few places where we want to use C++11 features, but these do
// not play nicely with the CUDA compiler. This file wraps them up so they can
// be compiled separately.
//
// See http://stackoverflow.com/a/21661547/591483 for more information.

#ifndef CXX11_H
#define CXX11_H

// uses <atomic>
#ifdef __cplusplus
extern "C" {
#endif
    void chunk_reset();
    unsigned chunk_get();
#ifdef __cplusplus
}
#endif

// uses std::uniform_int_distribution
class uniform_rng {
public:
    uniform_rng(int a, int b);
    // TODO: copy and move constructors (not used here)
    ~uniform_rng();

    // Generate one observation
    // int rand();
    int rand_from_range();
private:
    // PIMPL
    struct RngState;
    RngState *state;
};

#endif /* CXX11_H */
