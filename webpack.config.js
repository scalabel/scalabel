/* global module __dirname */
const CopyWebpackPlugin = require('copy-webpack-plugin');
const webpack = require('webpack');

let config = {
  devtool: '',
  entry: {
    create: __dirname + '/app/src/js/v1/create.js',
    image: __dirname + '/app/src/js/v1/image.index.js',
    image_v2: __dirname + '/app/src/js/entries/image.index.js',
    point_cloud: __dirname + '/app/src/js/v1/point_cloud/point_cloud.index.js',
    dashboard: __dirname + '/app/src/js/v1/dashboard.index.js',
    vendor: __dirname + '/app/src/js/v1/vendor.index.js',
  },
  output: {
    filename: '[name].js',
    path: __dirname + '/app/dist/js/',
  },
  plugins: [
    new webpack.ProvidePlugin({
      '$': 'jquery',
      'jQuery': 'jquery',
      'window.jQuery': 'jquery',
      'Popper': ['popper.js', 'default'],
    }),
    new CopyWebpackPlugin([
      {
        from: __dirname + '/app/src/annotation',
        to: __dirname + '/app/dist/annotation',
      },
      {
        from: __dirname + '/app/src/control',
        to: __dirname + '/app/dist/control',
      },
      {from: __dirname + '/app/src/css', to: __dirname + '/app/dist/css'},
      {from: __dirname + '/app/src/img', to: __dirname + '/app/dist/img'},
      {
        from: __dirname + '/app/src/index.html',
        to: __dirname + '/app/dist/index.html',
      },
      {
        from: __dirname + '/app/src/favicon.ico',
        to: __dirname + '/app/dist/favicon.ico',
      },
    ]),
  ],
  performance: {
    hints: false,
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /(node_modules|bower_components)/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env', '@babel/preset-flow'],
          },
        },
      },
    ],
  },
};

module.exports = (env /* : Object */, argv /* : Object */) => {
  if (argv.mode === 'development') {
    config.devtool = 'source-map';
  }

  return config;
};
