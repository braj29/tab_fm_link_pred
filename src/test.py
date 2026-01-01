from kubejobs.jobs import KubernetesJob

# Create a Kubernetes Job with a name, container image, and command
job = KubernetesJob(
    name="my-job",
    image="ubuntu:20.04",
    command=["/bin/bash", "-c", "echo 'Hello, World!'"],
)

# Generate the YAML configuration for the Job
print(job.generate_yaml())

# Run the Job on the Kubernetes cluster
job.run()

# Monitor the Job's status