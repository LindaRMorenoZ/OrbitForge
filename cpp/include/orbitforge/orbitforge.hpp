#pragma once

#include <cstddef>
#include <vector>

namespace orbitforge {

struct Vec2 {
    double x{};
    double y{};
    Vec2& operator+=(const Vec2& rhs);
    Vec2& operator-=(const Vec2& rhs);
};

Vec2 operator+(Vec2 lhs, const Vec2& rhs);
Vec2 operator-(Vec2 lhs, const Vec2& rhs);
Vec2 operator*(double scalar, Vec2 value);
double dot(const Vec2& lhs, const Vec2& rhs);

struct Body {
    double mass{};
    Vec2 position{};
    Vec2 velocity{};
};

struct State {
    std::vector<Body> bodies;
    double time{};
    void validate() const;
};

std::vector<Vec2> accelerations(
    const State& state, double gravitational_constant = 1.0, double softening = 0.0
);

void velocity_verlet_step(
    State& state, double dt, double gravitational_constant = 1.0,
    double softening = 0.0
);

double kinetic_energy(const State& state);
double potential_energy(
    const State& state, double gravitational_constant = 1.0, double softening = 0.0
);
double total_energy(
    const State& state, double gravitational_constant = 1.0, double softening = 0.0
);
Vec2 linear_momentum(const State& state);

}  // namespace orbitforge
