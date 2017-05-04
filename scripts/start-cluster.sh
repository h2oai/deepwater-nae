#!/bin/bash

# Start SSH
sudo /sbin/init

# Start nodes
for i in `tail -n +2 /etc/JARVICE/nodes`; do
   /opt/sssh $i /opt/make-flatfile.sh
   /opt/sssh $i /opt/start-deepwater.sh &
done

# Start Master
/opt/make-flatfile.sh
/opt/start-h2o3.sh &

# Change Nginx Redirect
sudo sed -e 's/8888/54321/' -i /etc/nginx/sites-enabled/default
sudo sed -e 's/8888/54321/' -i /etc/nginx/sites-enabled/notebook-site

# Start Notebook
/usr/local/bin/nimbix_notebook

# Start RStudio
sudo /etc/init.d/rstudio-server restart
