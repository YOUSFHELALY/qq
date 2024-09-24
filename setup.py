from setuptools import setup, find_packages

setup(
    name='quantumlib',  # اسم المكتبة
    version='0.1',  # الإصدار
    description='Quantum computing and machine learning library',
    author='Yousef Elhelaly',
    author_email='yousefelhelalyy@gmail.com',
    packages=find_packages(),  # العثور على جميع الباكجات الفرعية
    install_requires=['numpy'],  # المتطلبات الضرورية
)
