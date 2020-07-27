import * as fs from 'fs-extra'
import * as yaml from 'js-yaml'
import _ from 'lodash'
import * as yargs from 'yargs'
import { StorageType } from '../const/config'
import { CognitoConfig, ServerConfig } from '../types/config'
import * as defaults from './defaults'
import Logger from './logger'

/**
 * Initializes backend environment variables
 */
export async function readConfig (): Promise<ServerConfig> {
  /**
   * Creates config, using defaults for missing fields
   * Make sure user env come last to override defaults
   */

  // Read the config file name from argv
  const argv = yargs
    .options({
      config: {
        type: 'string', demandOption: true,
        describe: 'Config file path.'
      },
      dev: {
        type: 'boolean', default: false,
        describe: 'Turn on developer mode'
      }
    })
    .argv
  if (argv.dev) {
    Logger.setLogLevel('debug')
  }
  const configPath: string = argv.config
  const config = parseConfig(configPath)
  if (argv.dev) {
    /**
     * If command line dev is set to true,
     * it overrides the config file setting
     */
    config.mode.dev = true
  }
  await validateConfig(config)
  return config
}

/**
 * Load and parse the config file
 * @param configPath
 */
export function parseConfig (configPath: string): ServerConfig {
  // Load the config file
  const userConfig: Partial<ServerConfig> =
    yaml.load(fs.readFileSync(configPath, 'utf8'))
  const objectFields = ['redis', 'bot', 'storage', 'user', 'mode']

  const storage = _.clone(defaults.serverConfig.storage)

  // Check the deprecated fields for backward compatibility
  if (userConfig.data) {
    storage.data = userConfig.data
  }
  if (userConfig.itemDir) {
    storage.itemDir = userConfig.itemDir
  }
  if (userConfig.database) {
    storage.type = userConfig.database
  }
  userConfig.storage = storage

  // Set the default object fields
  objectFields.map((field) => {
    if (_.has(userConfig, field)) {
      _.set(userConfig, field, {
        ..._.get(defaults.serverConfig, field),
        ..._.get(userConfig, field)
      })
    }
  })

  const fullConfig = {
    ...defaults.serverConfig,
    ...userConfig
  }
  return fullConfig
}

/**
 * Validate cognito config
 * @param cognito
 */
function validateCognitoConfig (cognito: CognitoConfig | undefined) {
  if (cognito) {
    if (!_.has(cognito, 'region')) {
      throw new Error('Region missed in config ')
    }
    if (!_.has(cognito, 'userPool')) {
      throw new Error('User pool missed in config')
    }
    if (!_.has(cognito, 'clientId')) {
      throw new Error('Client id missed in config')
    }
    if (!_.has(cognito, 'userPoolBaseUri')) {
      throw new Error('User pool base uri missed in config')
    }
    if (!_.has(cognito, 'callbackUri')) {
      throw new Error('Call back uri missed in config')
    }
  } else {
    throw new Error('Cognito setting missed in config')
  }
}

/**
 * Validate server config.
 * Mainly focusing on user management
 *
 * @param {ServerConfig} config
 */
async function validateConfig (config: ServerConfig) {
  if (config.storage.type === StorageType.LOCAL) {
    if (config.storage.itemDir &&
        !(await fs.pathExists(config.storage.itemDir))) {
      Logger.info(`Item dir ${config.storage.itemDir} does not exist. Creating it`)
      fs.ensureDirSync(config.storage.itemDir)
    }
  }

  // Redis validation
  if (!(config.redis.timeForWrite + 1.5 < config.redis.timeout)) {
    throw new Error(`Redis timeForWrite must be at least 1.5 seconds earlier than redisTimeout
      to ensure that write occurs before value is erased`)
  }

  if (config.user.on) {
    validateCognitoConfig(config.cognito)
  }
}
