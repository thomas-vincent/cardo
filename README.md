[![Build Status](https://travis-ci.org/thomas-vincent/cardo.svg?branch=master)](https://travis-ci.org/thomas-vincent/cardo)

[![Coverage Status](https://coveralls.io/repos/github/thomas-vincent/cardo/badge.svg?branch=master)](https://coveralls.io/github/thomas-vincent/cardo?branch=master)

Description
-----------

Cardo takes a bunch of image files and organize them as a table in an svg
document.The positions of images in the table are determined by features
which are resolved from file names.

The main usage is automatic reporting for scientific data analysis.
When a process is repeated for different sets of input parameters, cardo
gathers results and displays them in an n-dimensional table.

For example, say the images are stored in the following folders
and subsolders:
```
`-- my_study
    |-- scenario1
    |   |-- experiment1
    |   |   `-- growth_profile.png
    |   |-- experiment2
    |   |   `-- growth_profile.png
    |   `-- experiment3
    |       `-- growth_profile.png
    `-- scenario2
        |-- experiment1
        |   `-- growth_profile.png
        |-- experiment2
        |   `-- growth_profile.png
        `-- experiment3
            `-- growth_profile.png
```

Here the features are: scenario and experiment. For each combination of feature
values, there is a growth profile picture.

Calling:

    $ cardo ./my_study

prints a SVG document representing the table:

|              |       scenario1           |     scenario2             |  
|:------------:|:-------------------------:|:-------------------------:|
| experiment1  | ![growth_profile][growth] | ![growth_profile][growth] |
| experiment2  | ![growth_profile][growth] | ![growth_profile][growth] |
| experiment3  | ![growth_profile][growth] | ![growth_profile][growth] |

[growth]: https://github.com/thomas-vincent/cardo/blob/master/doc/images/growth.png

The table layout can be customized:
```{r, engine='shell', count_lines}
cardo ./my_study --row_features=1 --col_features=2
```
where `--row_features=1` sets the first level of folders (scenario) as rows
and `--col_features=2` sets the second level of folders (experiment) as columns.

The following layout is produced:

|              |       experiment1         |     experiment1           |     experiment1           |  
|:------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| scenario1    | ![growth_profile][growth] | ![growth_profile][growth] | ![growth_profile][growth] |
| scenario2    | ![growth_profile][growth] | ![growth_profile][growth] | ![growth_profile][growth] |
             
             
Installation
------------

```{r, engine='shell', count_lines}
python setup.py install
```
