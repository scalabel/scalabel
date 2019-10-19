# Starting the http and socket servers
Run the following command:
````
node app/dist/js/main.js --config ./data/config.yml
````
# Enabling user collaboration
Enable sync in the config:
````
cp app/config/sync_config.yml data/config.yml
````
Run the same command as usual to start the server. Now you can open multiple instances of localhost in your browser, and they automatically sync.
