import * as fs from "fs-extra"
import * as yaml from "js-yaml"
import _ from "lodash"
import * as yargs from "yargs"

import { StorageType } from "../const/config"
import { CognitoConfig, ServerConfig } from "../types/config"
import * as defaults from "./defaults"
import Logger from "./logger"

/**
 * Initializes backend environment variables
 */
export async function readConfig(): Promise<ServerConfig> {
  /**
   * Creates config, using defaults for missing fields
   * Make sure user env come last to override defaults
   */

  // Read the config file name from argv
  const argv = yargs.options({
    config: {
      type: "string",
      demandOption: true,
      describe: "Config file path."
    },
    dev: {
      type: "boolean",
      default: false,
      describe: "Turn on developer mode"
    }
  }).argv
  if (argv.dev) {
    Logger.setLogLevel("debug")
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
 *
 * @param configPath
 */
export function parseConfig(configPath: string): ServerConfig {
  // Load the config file
  const userConfig: Partial<ServerConfig> = yaml.load(
    fs.readFileSync(configPath, "utf8")
  )

  // Check the deprecated fields for backward compatibility
  const storage = _.clone(defaults.serverConfig.storage)
  const http = _.clone(defaults.serverConfig.http)
  if (userConfig.port !== undefined) {
    http.port = userConfig.port
  }
  if (userConfig.data !== undefined) {
    storage.data = userConfig.data
  }
  if (userConfig.itemDir !== undefined) {
    storage.itemDir = userConfig.itemDir
  }
  if (userConfig.database !== undefined) {
    storage.type = userConfig.database
  }
  // Use the correct fields are set, still give them higher priority
  userConfig.http = { ...http, ...userConfig.http }
  userConfig.storage = { ...storage, ...userConfig.storage }

  // Set the default object fields
  _.keys(defaults.serverConfig).map((field) => {
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
 *
 * @param cognito
 */
function validateCognitoConfig(cognito: CognitoConfig | undefined): void {
  if (cognito !== undefined) {
    if (!_.has(cognito, "region")) {
      throw new Error("Region missed in config ")
    }
    if (!_.has(cognito, "userPool")) {
      throw new Error("User pool missed in config")
    }
    if (!_.has(cognito, "clientId")) {
      throw new Error("Client id missed in config")
    }
    if (!_.has(cognito, "userPoolBaseUri")) {
      throw new Error("User pool base uri missed in config")
    }
    if (!_.has(cognito, "callbackUri")) {
      throw new Error("Call back uri missed in config")
    }
  } else {
    throw new Error("Cognito setting missed in config")
  }
}

/**
 * Validate server config.
 * Mainly focusing on user management
 *
 * @param {ServerConfig} config
 */
async function validateConfig(config: ServerConfig): Promise<void> {
  if (config.storage.type === StorageType.LOCAL) {
    if (
      config.storage.itemDir !== "" &&
      !(await fs.pathExists(config.storage.itemDir))
    ) {
      Logger.info(
        `Item dir ${config.storage.itemDir} does not exist. Creating it`
      )
      fs.ensureDirSync(config.storage.itemDir)
    }
  }

  if (config.user.on) {
    validateCognitoConfig(config.cognito)
  }
}
