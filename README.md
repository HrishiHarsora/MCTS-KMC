# Atomistic Insights into MOCVD Growth of GaAs from a closed Loop Autonomous system combining Directed Reinforcement Learning and Kinetic Monte Carlo Simulations

<img width="4407" height="2524" alt="Picture1" src="https://github.com/user-attachments/assets/9da63d58-3da8-4342-8101-c6ce883a7074" />

Traditional approaches to model thin film deposition processes rely on frameworks at the macroscale (Computational Fluid Dynamics (CFD)) or microscale (Molecular Dynamics (MD), 
Density Functional Theory (DFT)) which are limited in their applicability:
- Microscale (DFT, MD): Due to the small length and time scales necessitated by the algorithms behind these frameworks, They cannot be used to model systems of large length and time scales. Modelling at such scales is necessary as it provides the unique opportunity to "bridge" length and time scales by feeding the output of macroscale simulations (like pressure/temperature variations) into these microscale frameworks thus allowing true multiscale modelling.  
- Macroscale (CFD): Although large systems can be modelled these frameworks are not able to resolve the process at the atomistic scale. This is important for two reasons: (a) A process like thin film deposition is a process rich in various event which occur on the scale of atoms and how they "behave" on the substrate affects the quality of the deposited film. Furthermore, (b) The parameters used to define a CFD simulation setup (such as sticking coefficient of precursors, etc) have to be derived from microscale simulations. 

Approaches at the mesoscale, specifically the Kinetic Monte Carlo (KMC) framework has the ability to not only model the process with atomistic precision but it can also be scaled to experimentally relevant length and time scales. The Bortz Kalos Lebowitz (BKL) algorithm constitutes the KMC framework.

## Scaling to larger length and time scales

Traditional microscale approaches, which model systems with atomic resolution, cannot be scaled up due to the **limitations imposed by their integrators.**

### Molecular Dynamics 
The framework of MD evolves the positions x(t) and velocities v(t) of the particles through time using the [velocity verlet](https://en.wikipedia.org/wiki/Verlet_integration) algorithm. The timestep $$\Delta t$$ is usually **chosen to be of the order of femtoseconds**. This choice is dictated by the need to resolve high-frequency atomic vibrations (such as bond stretches which are of the order of 10–100 fs) and to preserve numerical stability and energy conservation. 

Because of this small integration step, simulating even a single microsecond of physical time requires $10^9$ time steps. Moreover, MD calculations scale as $\mathcal{O}(N \log N)$ to $\mathcal{O}(N^2)$ with the number of atoms $N$, making it computationally prohibitive to simulate domains larger than a few hundred nanometers and timespans beyond hundreds of nanoseconds.

### Density Functional Theory (DFT)

While MD relies on empirical interatomic potentials, DFT solves the electronic structure of the system using quantum mechanics by solving the Kohn Sham equation. However, solving these equations requires diagonalizing matrices that scale cubically with the number of electrons—i.e., $\mathcal{O}(N^3)$ computational complexity. 

As a result, DFT calculations are limited to small unit cells of a few hundred atoms at $0\text{ K}$, rendering them incapable of modeling macroscopic time dynamics or mesoscale spatial phenomena. This limitation makes it impossible for DFT to simulate the accumulation of the material necessary to form realistic film thicknesses.

### Kinetic Monte Carlo (KMC)
KMC is a Monte Carlo technique which can be used to evolve a system in real time. 


Microscale approaches, due to their small timesteps, spend most of their computational effort resolving atomic vibrations within a single local basin of the potential energy landscape (PES), during which the configuration of the system does not meaningfully change. In the context of film growth, this level of accuracy is unnecessarily expensive. KMC circumvents this and allows actual film growth by evolving the system through discrete "jumps" between distinct basins of the PES, so that only those transitions which actually alter the configuration are simulated. This is precisely the origin of the speedup: rather than deriving the accessible transitions on the fly from an interatomic or pseudopotential, KMC works from a predefined catalogue of events that the system is permitted to undergo, and the evolution of the system is expressed purely as a function of these events.


## Optimizer - MCTS

Film deposition, as modelled by the KMC forward mode, depends on the reactions considered. The occurrence of reactions itself depends on the parameters used in the rate equations. Hence, the deposition behavior as modelled by KMC is a function of the rate parameters. In the GaAs KMC model developed in this work, the following reactions are considered:

| Event class | Reactions | Rate equation |
|---|---|---|
| Adsorption of MMGa or AsH ($j$), on a terrace or at a step ($s$) | R1, R2, R6, R7 | $R_{\mathrm{ADS}} = \dfrac{P_{j}}{\sqrt{2\pi M_{j} k_B T}} \, A_{\text{site}} \, S_{0(j,s)} \, \exp\!\left(-\dfrac{E_{a,\mathrm{chem}(j)}}{T}\right)$ |
| Surface decomposition (H off AsH, CH<sub>3</sub> off MMGa) | R3, R8 | $k_i = A_i \, T^{\beta_i} \exp\!\left(-\dfrac{E_{a,i}}{T}\right)$ |
| Desorption of adsorbates, their fragments, and film atoms held by two or three bonds | R4, R5, R9, R10, R11, R18 | $k_i = A_i \, T^{\beta_i} \exp\!\left(-\dfrac{E_{a,i}}{T}\right)$ |
| Incorporation — a Ga–As bond forms, on a terrace or at a step | R12–R17, R19, R20 | $k_i = A_i \, T^{\beta_i} \exp\!\left(-\dfrac{E_{a,i}}{T}\right)$ |

Altogether, there are 54 parameters which govern the rate of the 20 reactions considered in the KMC model here. Rather than computing these parameters and predicting the growth rate, we treat the parameter set as an unknown variable. Then, we search for a set of values that reproduces the experimental growth rate across the temperature range. Hence, the problem boils down to a search in a high dimensional parameter space for an optimal set. For this, we utilize the Monte Carlo Tree Search (MCTS) algorithm which is an extremely efficient at searching high dimensional spaces. It does so by growing a search tree in the space through 4 steps illustrated below:

<img width="3546" height="2234" alt="Picture1" src="https://github.com/user-attachments/assets/dcebfce2-8f06-44bb-b6e6-dcd95eec9b0b" />

1. Selection - Choosing a parent node to spawn children from
2. Expansion - Spawning a child node by biasing the parameters of its parent
3. Simulation - Scoring the child node using KMC
4. Backpropagation - carrying the score back up to the root and updating the score of each parent node in the way
   

