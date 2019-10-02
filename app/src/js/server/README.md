# Enabling user collaboration
Enable sync in the config:
````
cp app/config/sync_config.yml data/config.yml
````
Run the same commands as usual to start the http server. To start the node server that handles synchronization, open a separate terminal tab and run:
````
node app/dist/js/server.js --config ./data/config.yml
````
Now you can open multiple instances of localhost in your browser, and they automatically sync.
