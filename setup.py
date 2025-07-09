from setuptools import setup, find_packages
def get_requirements(file_path):
    """Read requirements from a file and return a list, ignoring comments and editable installs."""
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    return [
        req.strip()
        for req in requirements
        if req.strip() and not req.startswith('#') and not req.startswith('-e')
    ]
setup(
    name='mlprojects',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'mlprojects=mlprojects.cli:main'
        ]
    },
    author='Tanvi Kamath',
    author_email='tanvikamath22@gmail.com'
)