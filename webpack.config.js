/* global module __dirname process */
const CopyWebpackPlugin = require('copy-webpack-plugin');
const CircularDependencyPlugin = require('circular-dependency-plugin');
const webpack = require('webpack');

let config = {
  devtool: '',
  entry: {
    create: __dirname + '/app/src/js/entries/create.tsx',
    image: __dirname + '/app/src/js/v1/image.index.js',
    image_v2: __dirname + '/app/src/js/entries/image.index.js',
    worker: __dirname + '/app/src/js/entries/worker.tsx',
    admin: __dirname + '/app/src/js/entries/admin.tsx',
    point_cloud: __dirname + '/app/src/js/v1/point_cloud/point_cloud.index.js',
    speed_test: __dirname + '/app/src/js/dev/speed_test.js',
    dashboard: __dirname + '/app/src/js/entries/dashboard.tsx',
    vendor: __dirname + '/app/src/js/entries/vendor.tsx',
    label: __dirname + '/app/src/js/entries/label.index.ts',
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
    new CircularDependencyPlugin({
      // exclude detection of files based on a RegExp
      exclude: /node_modules/,
      // add errors to webpack instead of warnings
      failOnError: true,
      // allow import cycles that include an asynchronous import,
      // e.g. via import(/* webpackMode: "weak" */ './file.js')
      allowAsyncCycles: false,
      // set the current working directory for displaying module paths
      cwd: process.cwd(),
    }),
    new CopyWebpackPlugin([{
      from: __dirname + '/app/src/annotation',
      to: __dirname + '/app/dist/annotation',
    },
    {
      from: __dirname + '/app/src/control',
      to: __dirname + '/app/dist/control',
    },
    {
      from: __dirname + '/app/src/css',
      to: __dirname + '/app/dist/css',
    },
    {
      from: __dirname + '/app/src/img',
      to: __dirname + '/app/dist/img',
    },
    {
      from: __dirname + '/app/src/index.html',
      to: __dirname + '/app/dist/index.html',
    },
    {
      from: __dirname + '/app/src/favicon.ico',
      to: __dirname + '/app/dist/favicon.ico',
    },
    {
      from: __dirname + '/app/src/dev',
      to: __dirname + '/app/dist/dev',
    },
    ]),
  ],
  performance: {
    hints: false,
  },
  resolve: {
    // changed from extensions: [".js", ".jsx"]
    extensions: ['.ts', '.tsx', '.js', '.jsx'],
  },
  module: {
    rules: [{
      test: /\.t(s|sx)$/,
      use: {
        loader: 'awesome-typescript-loader',
      },
    },
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

let nodeExternals = require('webpack-node-externals');
let serverConfig = {
  target: 'node',
  externals: [nodeExternals()],
  devtool: '',
  entry: {
    main: __dirname + '/app/src/js/server/main.ts',
  },
  output: {
    filename: '[name].js',
    path: __dirname + '/app/dist/js/',
  },
  plugins: [
    new CircularDependencyPlugin({
      // exclude detection of files based on a RegExp
      exclude: /node_modules/,
      // add errors to webpack instead of warnings
      failOnError: true,
      // allow import cycles that include an asynchronous import,
      // e.g. via import(/* webpackMode: "weak" */ './file.js')
      allowAsyncCycles: false,
      // set the current working directory for displaying module paths
      cwd: process.cwd(),
    }),
  ],
  performance: {
    hints: false,
  },
  resolve: {
    extensions: ['.ts', '.tsx'],
  },
  module: {
    rules: [{
      test: /\.node|t(s|sx)$/,
      use: {
        loader: 'ts-loader',
      },
    }],
  },
  node: {
    __dirname: false,
    __filename: false,
  },
};

module.exports = (env /* : Object */, argv /* : Object */) => {
  if (argv.mode === 'development') {
    config.devtool = 'source-map';
    serverConfig.devtool = 'source-map';
  }

  return [config, serverConfig];
};
