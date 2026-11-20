#include "orbitforge/orbitforge.hpp"

#include <cmath>
#include <stdexcept>

namespace orbitforge {

Vec2& Vec2::operator+=(const Vec2& rhs) { x += rhs.x; y += rhs.y; return *this; }
Vec2& Vec2::operator-=(const Vec2& rhs) { x -= rhs.x; y -= rhs.y; return *this; }
Vec2 operator+(Vec2 lhs, const Vec2& rhs) { return lhs += rhs; }
Vec2 operator-(Vec2 lhs, const Vec2& rhs) { return lhs -= rhs; }
Vec2 operator*(double scalar, Vec2 value) { return {scalar * value.x, scalar * value.y}; }
double dot(const Vec2& lhs, const Vec2& rhs) { return lhs.x * rhs.x + lhs.y * rhs.y; }

void State::validate() const {
    if (bodies.empty()) throw std::invalid_argument("state requires at least one body");
    if (!std::isfinite(time)) throw std::invalid_argument("time must be finite");
    for (const auto& body : bodies) {
        if (!std::isfinite(body.mass) || body.mass <= 0.0)
            throw std::invalid_argument("body mass must be finite and positive");
        if (!std::isfinite(body.position.x) || !std::isfinite(body.position.y) ||
            !std::isfinite(body.velocity.x) || !std::isfinite(body.velocity.y))
            throw std::invalid_argument("body coordinates must be finite");
    }
}

std::vector<Vec2> accelerations(const State& state, double gravitational_constant,
                                double softening) {
    state.validate();
    if (!std::isfinite(gravitational_constant) || gravitational_constant <= 0.0)
        throw std::invalid_argument("G must be finite and positive");
    if (!std::isfinite(softening) || softening < 0.0)
        throw std::invalid_argument("softening must be finite and non-negative");

    std::vector<Vec2> result(state.bodies.size());
    const double eps2 = softening * softening;
    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        for (std::size_t j = i + 1; j < state.bodies.size(); ++j) {
            const Vec2 delta = state.bodies[j].position - state.bodies[i].position;
            const double r2 = dot(delta, delta) + eps2;
            if (r2 == 0.0)
                throw std::invalid_argument("coincident bodies require positive softening");
            const double scale = gravitational_constant / (r2 * std::sqrt(r2));
            result[i] += (state.bodies[j].mass * scale) * delta;
            result[j] -= (state.bodies[i].mass * scale) * delta;
        }
    }
    return result;
}

void velocity_verlet_step(State& state, double dt, double gravitational_constant,
                          double softening) {
    state.validate();
    if (!std::isfinite(dt) || dt <= 0.0)
        throw std::invalid_argument("dt must be finite and positive");
    const auto a0 = accelerations(state, gravitational_constant, softening);
    State next = state;
    for (std::size_t i = 0; i < state.bodies.size(); ++i)
        next.bodies[i].position += dt * state.bodies[i].velocity
                                + (0.5 * dt * dt) * a0[i];
    const auto a1 = accelerations(next, gravitational_constant, softening);
    for (std::size_t i = 0; i < state.bodies.size(); ++i)
        next.bodies[i].velocity += (0.5 * dt) * (a0[i] + a1[i]);
    next.time += dt;
    state = std::move(next);
}

double kinetic_energy(const State& state) {
    state.validate();
    double energy = 0.0;
    for (const auto& body : state.bodies)
        energy += 0.5 * body.mass * dot(body.velocity, body.velocity);
    return energy;
}

double potential_energy(const State& state, double gravitational_constant,
                        double softening) {
    state.validate();
    if (!std::isfinite(gravitational_constant) || gravitational_constant <= 0.0)
        throw std::invalid_argument("G must be finite and positive");
    if (!std::isfinite(softening) || softening < 0.0)
        throw std::invalid_argument("softening must be finite and non-negative");
    double energy = 0.0;
    const double eps2 = softening * softening;
    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        for (std::size_t j = i + 1; j < state.bodies.size(); ++j) {
            const Vec2 delta = state.bodies[j].position - state.bodies[i].position;
            const double distance = std::sqrt(dot(delta, delta) + eps2);
            if (distance == 0.0)
                throw std::invalid_argument("coincident bodies require positive softening");
            energy -= gravitational_constant * state.bodies[i].mass
                    * state.bodies[j].mass / distance;
        }
    }
    return energy;
}

double total_energy(const State& state, double gravitational_constant, double softening) {
    return kinetic_energy(state) + potential_energy(state, gravitational_constant, softening);
}

Vec2 linear_momentum(const State& state) {
    state.validate();
    Vec2 momentum{};
    for (const auto& body : state.bodies)
        momentum += body.mass * body.velocity;
    return momentum;
}

}  // namespace orbitforge
