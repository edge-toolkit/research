# Experimental orchestration controller

This controller is a Harbour-FL prototype for dynamic workload placement. It observes Kubernetes resource signals,
evaluates a bounded policy, and can move an eligible FL role between configured nodes. The code keeps observation,
policy, planning, and Kubernetes execution separated so those pieces can later be extracted into Edge Toolkit Core.

It is separate from the normal three-client startup path. The project's `mise run test` task runs its focused tests;
placement experiments are launched separately through the benchmark driver.

From the project directory, list the optional scenarios with
`.venv/bin/python -m experiments.scheduling.benchmark --list`. These experiments are not required for the resilience
reference suite, and their placement policy is not part of the shared recovery package.
