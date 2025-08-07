from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(
    name='image_recommendation_system',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A content-based image recommendation system',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/image-recommendation-system',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Graphics',
    ],
    keywords='image recommendation content-based filtering computer vision',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/image-recommendation-system/issues',
        'Source': 'https://github.com/yourusername/image-recommendation-system',
    },
)
