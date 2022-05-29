try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lines = (line.strip() for line in open(filename))
    return [line for line in lines if line and not line.startswith("#")]


reqs = parse_requirements("requirements_small_compat.txt")
dep_links = [url for url in reqs if "http" in url]
reqs = [req for req in reqs if "http" not in req]
reqs += [url.split("egg=")[-1] for url in dep_links if "egg=" in url]

setup(
    name="honeybee-comb-inferer",
    version="0.1.0",
    author="Ivan Matoshchuk",
    author_email="ivan.matoshchuk@gmail.com",
    url="https://github.com/IvanMatoshchuk/honeybee_cells_segmentation_inference",
    description="Inference pipeline for segmentation of a honey bee comb",
    install_requires=reqs,
    dependency_links=dep_links,
    packages=find_packages(),
    package_dir={"honeybee_comb_inferer": "honeybee_comb_inferer/"},
    # package_data={"honeybee_comb_inferer": ["config/config.yaml", "config/label_classes.json"]},
    # package_data={"honeybee-comb-inferer": ["*"]},
    # data_files=[("config-honeybee-comb", ["config/config.yaml"]), ("label-classes", ["data/label_classes.json"])],
)
