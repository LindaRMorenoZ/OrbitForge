#include "orbitforge/orbitforge.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

int main() {
    using namespace orbitforge;
    State state{{
        {2.0, {-1.0, 0.0}, {0.0, 0.0}},
        {1.0, {1.0, 0.0}, {0.0, 0.0}}
    }, 0.0};

    const auto acc = accelerations(state);
    const Vec2 net_force = state.bodies[0].mass * acc[0]
                         + state.bodies[1].mass * acc[1];
    assert(std::abs(net_force.x) < 1e-12);
    assert(std::abs(net_force.y) < 1e-12);

    const double e0 = total_energy(state, 1.0, 0.02);
    for (int i = 0; i < 100; ++i) velocity_verlet_step(state, 1e-4, 1.0, 0.02);
    assert(state.time > 0.0);
    assert(std::isfinite(total_energy(state, 1.0, 0.02)));
    assert(std::isfinite(e0));

    bool rejected = false;
    try { velocity_verlet_step(state, -0.1); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    return 0;
}
