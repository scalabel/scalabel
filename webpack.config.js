/* global module __dirname */
const CopyWebpackPlugin = require('copy-webpack-plugin');
const webpack = require('webpack');

let config = {
  entry: {
    create: __dirname + '/app/src/js/create.js',
    image: __dirname + '/app/src/js/image.index.js',
    point_cloud: __dirname + '/app/src/js/point_cloud/point_cloud.index.js',
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
        from: __dirname + '/app/src/js/thirdparty',
        to: __dirname + '/app/dist/js/thirdparty',
      },
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

module.exports = (env, argv) => {
  if (argv.mode === 'development') {
    config.devtool = 'source-map';
  }

  return config;
};
