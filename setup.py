import setuptools

# with open("./requirements.txt") as f:
#     required = f.read().splitlines()
#
# setuptools.setup(
#     name="batteries",
#     version="0.0.1",
#     description="Batteries (helpful functions) for pytorch (ha-ha-ha).",
#     url="https://github.com/jambo6/batteries",
#     author="James Morrill",
#     author_email="james.morrill.6@gmail.com",
#     packages=setuptools.find_packages(),
#     install_requires=required,
# )
#
setuptools.setup(setup_requires=["pbr>=2.0.0"], pbr=True)
