import * as bodyParser from 'body-parser'
import express, { Application } from 'express'
import * as formidable from 'express-formidable'
import { createServer } from 'http'
import socketio from 'socket.io'
import 'source-map-support/register'
import { BotManager } from './bot_manager'
import { Hub } from './hub'
import { Listeners } from './listeners'
import Logger from './logger'
import { getAbsoluteSrcPath, HTMLDirectories } from './path'
import { ProjectStore } from './project_store'
import { RedisClient } from './redis_client'
import { RedisPubSub } from './redis_pub_sub'
import { RedisStore } from './redis_store'
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

  // set up middleware
  app.use(listeners.loggingHandler)

  // set up static handlers for serving html
  // TODO: set up '/' endpoint
  for (const HTMLDir of HTMLDirectories) {
    app.use(express.static(
      getAbsoluteSrcPath(HTMLDir), { extensions: ['html'] }))
  }

  // set up static handlers for serving javascript
  app.use('/js', express.static(getAbsoluteSrcPath('/')))

  // set up static handlers for serving items to label
  app.use('/items', express.static(config.itemDir))

  // set up post/get handlers
  app.get(Endpoint.GET_PROJECT_NAMES,
    listeners.projectNameHandler.bind(listeners))
  app.get(Endpoint.EXPORT,
   listeners.getExportHandler.bind(listeners))

  app.post(Endpoint.POST_PROJECT, formidable(),
    listeners.postProjectHandler.bind(listeners))
  app.post(Endpoint.DASHBOARD, bodyParser.json(),
    listeners.dashboardHandler.bind(listeners))
}

/**
 * Make a publisher or subscriber for redis
 * Subscribers can't take other actions, so separate clients for pub and sub
 */
function makeRedisPubSub (config: ServerConfig): RedisPubSub {
  const client = new RedisClient(config)
  return new RedisPubSub(client)
}
/**
 * Main function for backend server
 */
async function main (): Promise<void> {
  // initialize config
  const config = readConfig()

  // initialize storage
  const storage = await makeStorage(config.database, config.data)

  // initialize redis- need separate clients for different roles
  const cacheClient = new RedisClient(config)
  const redisStore = new RedisStore(config, storage, cacheClient)
  const publisher = makeRedisPubSub(config)
  const subscriber = makeRedisPubSub(config)

  // initialize high level managers
  const projectStore = new ProjectStore(storage, redisStore)
  const userManager = new UserManager(projectStore)

  if (config.model) {
    const botManager = new BotManager(config, subscriber, redisStore)
    await botManager.listen()
  }

  // start http and socket io servers
  const app: Application = express()
  const httpServer = createServer(app)
  const io = socketio(httpServer)

  // set up http handlers
  startHTTPServer(config, app, projectStore, userManager)

  // set up socket.io handler
  const hub = new Hub(config, projectStore, userManager, publisher)
  await hub.listen(io)

  httpServer.listen(config.port)

  return
}

// TODO: Verify this is good promise handling
main().then().catch((error: Error) => {
  Logger.error(error)
})
