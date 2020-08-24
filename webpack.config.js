/* global module __dirname process */
const CopyWebpackPlugin = require('copy-webpack-plugin');
const CircularDependencyPlugin = require('circular-dependency-plugin');

let config = {
  devtool: '',
  entry: {
    create: __dirname + '/app/src/entries/create.tsx',
    worker: __dirname + '/app/src/entries/worker.tsx',
    admin: __dirname + '/app/src/entries/admin.tsx',
    // speed_test: __dirname + '/app/src/dev/speed_test.js',
    dashboard: __dirname + '/app/src/entries/dashboard.tsx',
    vendor: __dirname + '/app/src/entries/vendor.tsx',
    label: __dirname + '/app/src/entries/label.index.ts',
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
    new CopyWebpackPlugin([
      {
        from: __dirname + '/app/html',
        to: __dirname + '/app/dist/html',
      },
      {
        from: __dirname + '/app/css',
        to: __dirname + '/app/dist/css',
      },
      {
        from: __dirname + '/app/img',
        to: __dirname + '/app/dist/img',
      },
      {
        from: __dirname + '/app/dev',
        to: __dirname + '/app/dist/dev',
      },
    ]),
  ],
  performance: {
    hints: false,
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.jsx'],
  },
  module: {
    rules: [
      {
        test: /\.t(s|sx)$/,
        use: {
          loader: 'awesome-typescript-loader',
        },
      }
    ],
  },
};

let nodeExternals = require('webpack-node-externals');
let serverConfig = {
  target: 'node',
  externals: [nodeExternals()],
  devtool: '',
  entry: {
    main: __dirname + '/app/src/server/main.ts',
  },
  output: {
    filename: '[name].js',
    path: __dirname + '/app/dist/',
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
