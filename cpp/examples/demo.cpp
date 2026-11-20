#include "orbitforge/orbitforge.hpp"

#include <iomanip>
#include <iostream>

int main() {
    orbitforge::State state{{
        {1.0, {-0.5, 0.0}, {0.0, -0.5}},
        {1.0, {0.5, 0.0}, {0.0, 0.5}}
    }, 0.0};

    const double initial_energy = orbitforge::total_energy(state, 1.0, 0.01);
    for (int i = 0; i < 1000; ++i)
        orbitforge::velocity_verlet_step(state, 0.001, 1.0, 0.01);

    std::cout << std::fixed << std::setprecision(8)
              << "Final time: " << state.time << "\n"
              << "Final total energy: " << orbitforge::total_energy(state, 1.0, 0.01)
              << "\nRelative energy change: "
              << (orbitforge::total_energy(state, 1.0, 0.01) - initial_energy)
                 / initial_energy << "\n";
    return 0;
}
