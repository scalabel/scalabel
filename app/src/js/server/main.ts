import * as child from 'child_process'
import express, { Application, NextFunction, Request, Response } from 'express'
import * as formidable from 'express-formidable'
import { createServer } from 'http'
import socketio from 'socket.io'
import 'source-map-support/register'
import { removeListItems } from '../functional/util'
import { BotManager } from './bot_manager'
import Callback from './controller/callback'
import { Hub } from './hub'
import { Listeners } from './listeners'
import Logger from './logger'
import auth from './middleware/cognitoAuth'
import errorHandler from './middleware/errorHandler'
import { getAbsoluteSrcPath, getRedisConf, HTMLDirectories } from './path'
import { ProjectStore } from './project_store'
import { RedisClient } from './redis_client'
import { RedisPubSub } from './redis_pub_sub'
import { RedisStore } from './redis_store'
import { Storage, STORAGE_FOLDERS, StorageStructure } from './storage'
import { Endpoint, ServerConfig } from './types'
import { UserManager } from './user_manager'
import { makeStorage, readConfig } from './util'

/**
 * Sets up http handlers
 */
function startHTTPServer (
  config: ServerConfig, app: Application,
  projectStore: ProjectStore, userManager: UserManager) {
  const listeners = new Listeners(projectStore, userManager)

  // Set up middleware
  app.use(listeners.loggingHandler)

  // Set up static handlers for serving html
  // TODO: set up '/' endpoint
  for (const HTMLDir of HTMLDirectories) {
    app.use(express.static(
      getAbsoluteSrcPath(HTMLDir), { extensions: ['html'] }))
  }

  // Set up static handlers for serving javascript
  app.use('/js', express.static(getAbsoluteSrcPath('/')))

  // Set up static handlers for serving items to label
  app.use('/items', express.static(config.itemDir))

  const authMiddleWare =
    config.userManagement ?
      auth(config) :
      (_req: Request, _res: Response, next: NextFunction) => next()

  app.set('views', getAbsoluteSrcPath('../control'))
  app.set('view engine', 'ejs')

  app.use(Endpoint.CALLBACK,
    new Callback(config).router)

  // Set up post/get handlers
  app.get(Endpoint.GET_PROJECT_NAMES, authMiddleWare,
    listeners.projectNameHandler.bind(listeners))
  app.get(Endpoint.EXPORT, authMiddleWare,
   listeners.getExportHandler.bind(listeners))

  app.post(Endpoint.POST_PROJECT, authMiddleWare, formidable(),
    listeners.postProjectHandler.bind(listeners))
  app.post(Endpoint.POST_PROJECT_INTERNAL, authMiddleWare, express.json(),
    listeners.postProjectInternalHandler.bind(listeners))
  app.post(Endpoint.POST_TASKS, authMiddleWare, express.json(),
    listeners.postTasksHandler.bind(listeners))
  app.post(Endpoint.DASHBOARD, authMiddleWare, express.json(),
    listeners.dashboardHandler.bind(listeners))
  app.use(errorHandler(config))
}

/**
 * Make a publisher or subscriber for redis
 * Subscribers can't take other actions, so separate clients for pub and sub
 */
function makeRedisPubSub (config: ServerConfig): RedisPubSub {
  const client = new RedisClient(config.redis)
  return new RedisPubSub(client)
}

/**
 * Starts a bot manager if config says to
 */
async function makeBotManager (
  config: ServerConfig, subscriber: RedisPubSub, cacheClient: RedisClient) {
  if (config.bot.on) {
    const botManager = new BotManager(config.bot, subscriber, cacheClient)
    await botManager.listen()
  }
}

/**
 * Launch the redis server
 */
async function launchRedisServer (config: ServerConfig) {
  let redisDir = './'
  if (config.database === 'local') {
    redisDir = config.data
  }

  const redisProc = child.spawn('redis-server', [
    getRedisConf(),
    '--port', `${config.redis.port}`,
    '--bind', '127.0.0.1',
    '--dir', redisDir,
    '--protected-mode', 'yes']
  )
  redisProc.stdout.on('data', (data) => {
    process.stdout.write(data)
  })

  redisProc.stderr.on('data', (data) => {
    process.stdout.write(data)
  })
}

/**
 * Start HTTP and socket io servers
 */
async function startServers (
  config: ServerConfig, projectStore: ProjectStore,
  userManager: UserManager, publisher: RedisPubSub) {
  const app: Application = express()
  const httpServer = createServer(app)
  const io = socketio(httpServer)

  // Set up http handlers
  startHTTPServer(config, app, projectStore, userManager)

  // Set up socket.io handler
  const hub = new Hub(config, projectStore, userManager, publisher)
  await hub.listen(io)

  httpServer.listen(config.port)
}

/**
 * Check wether there is legacy project folders in storage
 * @param storage
 */
async function checkLegacyProjectFolders (storage: Storage) {
  let folders = await storage.listKeys('', true)
  folders = removeListItems(
    folders, STORAGE_FOLDERS)
  if (folders.length > 0) {
    const cmd = `cd ${storage.dataDir}; mv ${''.concat(...folders.map((f) => f + ' '))} ${StorageStructure.PROJECT}; cd ../..`
    Logger.info(`Detected legacy project names [${folders.toString()}] ` +
    `under the scalabel folder. ` +
    `Please move them to the ${StorageStructure.PROJECT}/ folder and ` +
    `relaunch scalabel. You can run command "${cmd}" to move the folders.`)
    process.exit(1)
  }
}

/**
 * Main function for backend server
 */
async function main () {
  // Initialize config
  const config = await readConfig()

  // Initialize storage
  const storage = await makeStorage(config.database, config.data)

  await checkLegacyProjectFolders(storage)

  // Start the redis server
  await launchRedisServer(config)

  /**
   * Connect to redis server with clients
   * Need separate clients for different roles
   */
  const cacheClient = new RedisClient(config.redis)
  const redisStore = new RedisStore(config.redis, storage, cacheClient)
  const publisher = makeRedisPubSub(config)
  const subscriber = makeRedisPubSub(config)

  // Initialize high level managers
  const projectStore = new ProjectStore(storage, redisStore)
  const userManager = new UserManager(projectStore, config.userManagement)
  await userManager.clearUsers()

  await makeBotManager(config, subscriber, cacheClient)
  await startServers(config, projectStore, userManager, publisher)

  return
}

// TODO: Verify this is good promise handling
main().then().catch((error: Error) => {
  Logger.error(error)
})
