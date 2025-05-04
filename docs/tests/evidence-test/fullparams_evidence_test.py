import os
import json
import argparse
import bilby
import numpy as np
import bilby.core.utils as bcu
from bilby.gw.prior import PriorDict
from bilby.core.prior import Uniform


class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self, mean, cov):
        super().__init__(parameters={f"amp{i+1}": None for i in range(len(mean))})
        self.mean = mean
        self.cov = cov
        self.inv_sigma = np.linalg.inv(cov)
        self.det_sigma = np.linalg.det(cov)
        self.d = len(mean)

    def log_likelihood(self):
        amps = np.array([self.parameters[f"amp{i+1}"] for i in range(self.d)])
        diff = amps - self.mean
        return -0.5 * (diff @ self.inv_sigma @ diff)


def main(lmns, version):

    # Configuration
    label = "Gaussian_example"
    outdir = f"results/evidence-test/Case-{lmns}/run-{version}"
    bcu.setup_logger(outdir=outdir, label=label)

    # Load Fisher matrix results
    file_path = f"./results/evidence-test/fisher_matrix/Fisher_matrix_{lmns}.jsonl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Please run `fisher_matrix_calculations.py` first to obtain the Fisher matrix."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        record = json.loads(f.readline().strip())

    mean = np.array(record["MLE"])
    cov = np.array(record["Cov"])

    # Initialize likelihood and priors
    likelihood = SimpleGaussianLikelihood(mean=mean, cov=cov)
    priors = PriorDict()
    for i in range(1, len(mean) + 1):
        priors[f"amp{i}"] = Uniform(
            name=f"amp{i}", minimum=-5, maximum=5, latex_label=f"$amp{i}$"
        )

    # Run sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=2000,
        outdir=outdir,
        label=label,
        queue_size=16,
    )

    bcu.logger.info(f"The evidence is: {result.log_evidence + np.log(10**len(mean))}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Gaussian likelihood example")
    parser.add_argument("--lmns", type=str, required=True)
    parser.add_argument("--version", type=int, default=0, help="Run version number")
    args = parser.parse_args()

    lmns = args.lmns
    version = args.version
    main(lmns, version)
