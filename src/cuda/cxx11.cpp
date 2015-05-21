#include <atomic>
#include <random>

#include "cxx11.h"

using namespace std;

std::atomic<unsigned> chunk_id;

void chunk_reset(void)
{
	chunk_id = 0;
}

unsigned chunk_get(void)
{
	return chunk_id.fetch_add(1);
}

struct uniform_rng::RngState
{
    std::mt19937* gen;
    std::uniform_int_distribution<int> *distribution;
};

uniform_rng::uniform_rng(int a, int b)
{
    state = new uniform_rng::RngState();

    // TODO: you could also look at better seeding algorithms; right now you're getting the same numbers on every instantiation
    // http://www.cplusplus.com/reference/random/mersenne_twister_engine/seed/

    std::random_device rd; // only used for seeding the PRNG

    state->gen = new std::mt19937(rd());
    state->distribution = new std::uniform_int_distribution<int>(a, b);
}

int uniform_rng::rand_from_range()
{
    // TODO: this seems to be in poor style
    std::mt19937& gen = *state->gen;
    std::uniform_int_distribution<int>& dist = *state->distribution;

    return dist(gen);
}

uniform_rng::~uniform_rng()
{
    delete state->distribution;
    delete state->gen;
    delete state;
}
