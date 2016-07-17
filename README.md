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
   

Here the features are: scenario and experiment. For each combination of feature
values, there is a growth profile picture.

Calling:

    $ cardo ./my_study

produces the svg document './my_study.svg' with the following content:


                    scenario1                scenario2        
                                                              
             +----------------------+ +----------------------+
             |                      | |                      |
             |                      | |                      |
             |  growth_profile.png  | |  growth_profile.png  |
experiment1  |  for scenario1 and   | |  for scenario2 and   |
             |  experiment1         | |  experiment1         |
             |                      | |                      |
             |                      | |                      |
             +----------------------+ +----------------------+

             +----------------------+ +----------------------+
             |                      | |                      |
             |                      | |                      |
             |  growth_profile.png  | |  growth_profile.png  |
experiment2  |  for scenario1 and   | |  for scenario2 and   |
             |  experiment2         | |  experiment2         |
             |                      | |                      |
             |                      | |                      |
             +----------------------+ +----------------------+

             +----------------------+ +----------------------+
             |                      | |                      |
             |                      | |                      |
             |  growth_profile.png  | |  growth_profile.png  |
experiment3  |  for scenario1 and   | |  for scenario2 and   |
             |  experiment3         | |  experiment3         |
             |                      | |                      |
             |                      | |                      |
             +----------------------+ +----------------------+


The table layout can be customized:
   $ cardo ./my_study --row_features=1 --col_features=2

where --row_features=1 sets the first level of folders (scenario) as rows
and --col_features=2 sets the second level of folders (experiment) as columns.

The following layout is produced:

                    experiment1              experiment2             experiment3  
                                                              
             +----------------------+ +----------------------+ +----------------------+ 
             |                      | |                      | |                      | 
             |                      | |                      | |                      | 
             |  growth_profile.png  | |  growth_profile.png  | |  growth_profile.png  | 
scenario1    |  for scenario1 and   | |  for scenario1 and   | |  for scenario1 and   | 
             |  experiment1         | |  experiment2         | |  experiment3         | 
             |                      | |                      | |                      | 
             |                      | |                      | |                      | 
             +----------------------+ +----------------------+ +----------------------+
                                                               
             +----------------------+ +----------------------+ +----------------------+
             |                      | |                      | |                      |
             |                      | |                      | |                      |
             |  growth_profile.png  | |  growth_profile.png  | |  growth_profile.png  |
scenario2    |  for scenario2 and   | |  for scenario1 and   | |  for scenario2 and   |
             |  experiment1         | |  experiment2         | |  experiment3         |
             |                      | |                      | |                      |
             |                      | |                      | |                      |
             +----------------------+ +----------------------+ +----------------------+
             
             
Installation
------------

python setup.py install

