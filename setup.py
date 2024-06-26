from setuptools import setup, find_packages

setup(packages=find_packages(where="src"),
      package_dir={"": "src"},
      package_data={"mllibs": ["corpus/*.txt",
                               "corpus/*.csv",
                               "nlp/*.json",
                               "eda/*.json",
                               "libop/*.json",
                               "stats/*.json",
                               "signal/*.json",
                               "pd/*.json",
                               "ml/*.json",
                               "*.json",
                               "models/*.pickle"]
      })


