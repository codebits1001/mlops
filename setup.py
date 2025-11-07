from setuptools import find_packages, setup
from typing import List

# Lets create the function that is get_requirements
hyphen_e = '-e .'

def get_requirements(file_name: str ) ->List[str]:


    # create the list that stores the packages to install

    require = {}

    # open the file
    with open(file_name) as f:
        # read the file
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        # escape the hypen part as well 

        if hyphen_e in requirements:
            requirements.remove(hyphen_e)
        return requirements








setup(
    name = 'mlops_project_pipeline',
    version = '0.01',
    author = 'Braino_G',
    author_email = 'codebits1001@icloud.com',
    packages = find_packages(),
    install_requries = get_requirements('requirements.txt')

)