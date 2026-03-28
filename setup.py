#!/usr/bin/env python3

import os
import glob
from setuptools import setup, find_packages

package_name = 'river_lane_pilot'

# 递归查找所有子包
def find_all_packages():
    """查找所有子包"""
    packages = find_packages()
    
    # 手动添加可能遗漏的包
    additional_packages = [
        'river_lane_pilot.utils',
        'river_lane_pilot.perception', 
        'river_lane_pilot.planning',
        'river_lane_pilot.control'
    ]
    
    for pkg in additional_packages:
        if pkg not in packages:
            packages.append(pkg)
    
    return packages

# 查找数据文件
def find_data_files():
    """查找数据文件"""
    data_files = []
    
    # 配置文件
    config_files = glob.glob('config/*')
    if config_files:
        data_files.append(('share/' + package_name + '/config', config_files))
    
    # 资源文件
    resource_files = glob.glob('resource/*')
    if resource_files:
        data_files.append(('share/' + package_name + '/resource', resource_files))
    
    # 文档文件
    doc_files = ['README.md']
    existing_doc_files = [f for f in doc_files if os.path.exists(f)]
    if existing_doc_files:
        data_files.append(('share/' + package_name + '/doc', existing_doc_files))
    
    # 模型文件
    model_files = glob.glob('models/*')
    if model_files:
        data_files.append(('share/' + package_name + '/models', model_files))
    
    return data_files

# 查找脚本文件
def find_scripts():
    """查找可执行脚本"""
    scripts = []
    
    # scripts目录下的Python文件
    script_files = glob.glob('scripts/*.py')
    for script in script_files:
        if os.path.isfile(script):
            scripts.append(script)
    
    return scripts

# 读取requirements.txt
def read_requirements():
    """读取requirements.txt文件"""
    requirements = []
    
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 处理不同的包名格式
                    if '>=' in line or '==' in line or '<=' in line:
                        requirements.append(line)
                    else:
                        requirements.append(line)
    except FileNotFoundError:
        print("Warning: requirements.txt not found")
    
    return requirements

# 读取README文件作为长描述
def read_long_description():
    """读取README文件"""
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "River Lane Pilot - Autonomous USV Navigation System"

setup(
    name=package_name,
    version='1.0.0',
    description='自主驾驶水上无人船导航系统 - Autonomous USV Navigation System',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    
    # 作者信息
    author='River Pilot Development Team',
    author_email='developer@river-pilot.com',
    maintainer='River Pilot Team',
    maintainer_email='maintainer@river-pilot.com',
    
    # 项目信息
    url='https://github.com/river-pilot/river_lane_pilot',
    download_url='https://github.com/river-pilot/river_lane_pilot/releases',
    license='MIT',
    
    # 包信息
    packages=find_all_packages(),
    package_dir={'': '.'},
    
    # 数据文件
    data_files=find_data_files(),
    
    # 依赖
    install_requires=read_requirements(),
    
    # Python版本要求
    python_requires='>=3.8',
    
    # 入口点 (脚本和工具)
    entry_points={
        'console_scripts': [
            # 工具脚本
            'config_tool = river_lane_pilot.tools.config_tool:main',
            'model_converter = river_lane_pilot.tools.model_converter:main',
            'data_recorder = river_lane_pilot.tools.data_recorder:main',
            'performance_analyzer = river_lane_pilot.tools.performance_analyzer:main',
        ],
    },
    
    # 分类信息
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: System :: Hardware :: Hardware Drivers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    # 关键词
    keywords=[
        'autonomous navigation', 'USV', 'unmanned surface vehicle',
        'computer vision', 'deep learning', 'segformer', 'pure pursuit',
        'maritime robotics', 'lane detection', 'path planning', 'control',
        'NVIDIA Jetson', 'river navigation', 'robotics'
    ],
    
    # 项目URL
    project_urls={
        'Bug Reports': 'https://github.com/river-pilot/river_lane_pilot/issues',
        'Source': 'https://github.com/river-pilot/river_lane_pilot',
        'Documentation': 'https://river-pilot.readthedocs.io/',
        'Funding': 'https://github.com/sponsors/river-pilot',
    },
    
    # 额外文件
    include_package_data=True,
    
    # 平台特定的依赖
    extras_require={
        'jetson': [
            'jetson-gpio>=2.0.0',
            'jetson-stats>=4.0.0',
        ],
        'cuda': [
            'tensorrt>=8.0.0',
            'pycuda>=2021.1',
        ],
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0', 
            'black>=21.0.0',
            'flake8>=3.8.0',
            'mypy>=0.800',
            'pre-commit>=2.10.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
            'myst-parser>=0.15.0',
        ],
        'visualization': [
            'foxglove-websocket>=0.0.8',
        ]
    },
    
    # 测试套件
    test_suite='tests',
    
    # zip安全
    zip_safe=False,
    
    # 脚本文件
    scripts=find_scripts(),
)