"""
All synthetic datasets
"""

import numpy as np
from tqdm import tqdm


class springMassSystem():
    def __init__(
            self,
            n_masses=8,
            arena_size=1.0,
            force_field_proportion=0.85,
            spring_constant=0.5,
            damping_coef=0.07,
            init_vel_norm=1.0,
            time=100000,
            dt=1e-4,
            rng=np.random.default_rng()
    ):
        """
        A vanilla mass spring system with a 'force field' that pushes masses 
        away from the boundary.

        :param n_masses: Number of masses
        :param arena_size: size of the arena
        :param force_field_proportion: proportion of the arena that has no
          force field
        :param spring_constant: spring constant for all springs
        :param init_loc_std: standard deviation of the initial positions of
          the masses
        :param init_vel_norm: norm of the initial velocity vectors
        :param time: how long to simulate the system for
        :param dt: time step
        :param seed: random seed
        """
        self.n_masses = n_masses
        self.arena_size = arena_size
        self.spring_constant = spring_constant
        self.damping_coef = damping_coef
        self.init_vel_norm = init_vel_norm
        self.time = time
        self.dt = dt
        self.rng = rng
        self.eps = 1e-16

        # calculating distance where force field takes effect
        self.force_field_size = arena_size * force_field_proportion

    
    def _init_masses(self):
        """
        Initialize the system masses.
        
        :return init_pos: initial position of the masses
        :return init_vel: initial velocity of the masses
        """
        # initial mass positions
        theta = self.rng.uniform(0.0, 2 * np.pi, size=self.n_masses)
        u = self.rng.uniform(0.0, 1.0, size=self.n_masses)
        r = self.force_field_size * np.sqrt(u)
        init_pos = np.array([r * np.cos(theta), r * np.sin(theta)])

        # initial mass velocities
        init_vel = self.rng.standard_normal((2, self.n_masses))
        init_vel = init_vel / (np.linalg.norm(init_vel, axis=0) + self.eps)
        init_vel = init_vel * self.init_vel_norm

        return init_pos, init_vel
    
    def _init_springs(self):
        """
        Initialize the system springs.
        
        :return A: Adjacency matrix of the spring mass system
        """
        # create a random adjacency matrix
        A = self.rng.binomial(
            n=1,
            p=0.5,
            size=(self.n_masses, self.n_masses)
        )

        # Make it undirected with no self edges
        A = np.triu(A)
        A = A.T + A
        np.fill_diagonal(A, 0)

        return A

    def simulate(self):
        """
        Docstring for simulate
        
        :return p_t: Position of masses over time
        :return v_t: Velocity of masses over time
        :return A: Edges between masses
        """
        # initialize system
        pos_next, vel_next = self._init_masses()
        A = self._init_springs()
        p_t = np.zeros((2, self.n_masses, self.time+1))
        v_t = np.zeros((2, self.n_masses, self.time+1))
        p_t[:,:,0] = pos_next
        v_t[:,:,0] = vel_next
        A_t = np.repeat(A[:, :, None], repeats=self.time+1, axis=-1)

        # simulate
        for t in tqdm(range(self.time), desc='simulating'):
            # spring force calculation
            dist_x = pos_next[0, :][:, None] - pos_next[0, :][None, :]
            dist_y = pos_next[1, :][:, None] - pos_next[1, :][None, :]
            f_x = np.sum(-self.spring_constant * A * dist_x, axis=1)
            f_y = np.sum(-self.spring_constant * A * dist_y, axis=1)
            f_spring = np.stack([f_x, f_y], axis=0)

            # force field force calculation (magnitude exponential wrt to distance from center)
            pos_norm = np.linalg.norm(pos_next, axis=0)
            f_force_field = np.where(
                pos_norm > self.force_field_size,
                -pos_next,
                0.
            )
            f_force_field = (f_force_field / (np.linalg.norm(f_force_field, axis=0) + self.eps)) * np.exp(np.linalg.norm(f_force_field, axis=0))

            # damping force
            f_damping = -self.damping_coef * vel_next

            # totl force
            f_tot = f_spring + f_force_field + f_damping

            # update position and velocity and store
            vel_next += self.dt * f_tot
            pos_next += self.dt * vel_next

            p_t[:,:,t+1] = pos_next
            v_t[:,:,t+1] = vel_next

        return p_t, v_t, A_t


class springMassDynamicEdges(springMassSystem):
    def __init__(
            self,
            fov_angle=115,
            n_masses=8,
            arena_size=1.0,
            force_field_proportion=0.85,
            spring_constant=0.5,
            damping_coef=0.07,
            init_vel_norm=1.0,
            time=100000,
            dt=1e-4,
            rng=np.random.default_rng()
    ):
        """
        A mass spring system with a 'force field' that pushes masses 
        away from the boundary and has dynamicly changing spring positions
        over time.

        :param fov_angle: angle to the left and right the mass can "see"
        :param n_masses: Number of masses
        :param arena_size: size of the arena
        :param force_field_proportion: proportion of the arena that has no
          force field
        :param spring_constant: spring constant for all springs
        :param init_loc_std: standard deviation of the initial positions of
          the masses
        :param init_vel_norm: norm of the initial velocity vectors
        :param time: how long to simulate the system for
        :param dt: time step
        :param seed: random seed
        """
        super().__init__(
            n_masses=n_masses,
            arena_size=arena_size,
            force_field_proportion=force_field_proportion,
            spring_constant=spring_constant,
            damping_coef=damping_coef,
            init_vel_norm=init_vel_norm,
            time=time,
            dt=dt,
            rng=rng
        )
        self.fov_angle = fov_angle

    def _calculate_edges(self, pos, vel):
        # get relative distances
        rel_dists = pos[:, None, :] - pos[:, :, None]
        dist_norm = np.linalg.norm(rel_dists, axis=0)
        rel_dists = rel_dists / (dist_norm + self.eps)

        # normalize velocity
        vel = vel / (np.linalg.norm(vel, axis=0) + self.eps)
        vel_perp = np.stack([-vel[1, :], vel[0, :]], axis=0)

        # calculate pairwise dot products from perpendicular headings
        dot_prods = (vel_perp[:, :, None] * rel_dists).sum(axis=0)

        # Adjacency matrix is any within fov_angle degrees of heading
        fov_thresh = np.cos(self.fov_angle * np.pi/180 / 2)
        fov_mask = (np.abs(dot_prods) >= fov_thresh) & (dist_norm > 0.0)
        A = fov_mask | fov_mask.T  # directed
        np.fill_diagonal(A, 0)

        return A

    def simulate(self):
        # initialize system
        pos_next, vel_next = self._init_masses()
        A = self._calculate_edges(pos_next, vel_next)
        p_t = np.zeros((2, self.n_masses, self.time+1))
        v_t = np.zeros((2, self.n_masses, self.time+1))
        A_t = np.zeros((self.n_masses, self.n_masses, self.time+1))
        p_t[:,:,0] = pos_next
        v_t[:,:,0] = vel_next
        A_t[:,:,0] = A

        # simulate
        for t in tqdm(range(self.time), desc='simulating'):
            # spring force calculation
            dist_x = pos_next[0, :][:, None] - pos_next[0, :][None, :]
            dist_y = pos_next[1, :][:, None] - pos_next[1, :][None, :]
            f_x = np.sum(-self.spring_constant * A * dist_x, axis=1)
            f_y = np.sum(-self.spring_constant * A * dist_y, axis=1)
            f_spring = np.stack([f_x, f_y], axis=0)

            # force field force calculation
            pos_norm = np.linalg.norm(pos_next, axis=0)
            f_force_field = np.where(
                pos_norm > self.force_field_size,
                -pos_next,
                0.
            )
            f_force_field = (f_force_field / (np.linalg.norm(f_force_field, axis=0) + self.eps)) * np.exp(np.linalg.norm(f_force_field, axis=0))

            # damping force
            f_damping = -self.damping_coef * vel_next

            # total force
            f_tot = f_spring + f_force_field + f_damping

            # update position and velocity and store
            vel_next += self.dt * f_tot
            pos_next += self.dt * vel_next
            A = self._calculate_edges(pos_next, vel_next)

            p_t[:,:,t+1] = pos_next
            v_t[:,:,t+1] = vel_next
            A_t[:,:,t+1] = A

        return p_t, v_t, A_t
