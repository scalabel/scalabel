/* global module __dirname process */
const CopyWebpackPlugin = require('copy-webpack-plugin');
const CircularDependencyPlugin = require('circular-dependency-plugin');
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');

let config = {
  devtool: false,
  entry: {
    create: __dirname + '/app/src/entries/create.tsx',
    worker: __dirname + '/app/src/entries/worker.tsx',
    admin: __dirname + '/app/src/entries/admin.tsx',
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
    new CopyWebpackPlugin({
      patterns: [
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
    ]}),
    // used for type checking when `transpile: true` for ts-loader
    // see https://github.com/TypeStrong/ts-loader#transpileonly
    new ForkTsCheckerWebpackPlugin()
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
          loader: 'ts-loader',
          // Temporary fix to avoid memory errors
          options: { transpileOnly: true }
        },
      },
      {
        test: /\.m?js$/,
        exclude: /(node_modules|bower_components)/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader']
      }
    ],
  },
};

let nodeExternals = require('webpack-node-externals');
let serverConfig = {
  target: 'node',
  externals: [nodeExternals()],
  devtool: false,
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
    // used for type checking when `transpile: true` for ts-loader
    // see https://github.com/TypeStrong/ts-loader#transpileonly
    new ForkTsCheckerWebpackPlugin()
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
        options: { transpileOnly: true }
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
