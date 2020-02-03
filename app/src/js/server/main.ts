import * as bodyParser from 'body-parser'
import express, { Application } from 'express'
import * as formidable from 'express-formidable'
import { createServer } from 'http'
import * as socketio from 'socket.io'
import { startSocketServer } from './hub'
import { Listeners } from './listeners'
import { getAbsoluteSrcPath, HTMLDirectories } from './path'
import { ProjectStore } from './project_store'
import { Endpoint } from './types'
import { UserManager } from './user_manager'
import { makeEnv, makeStorage } from './util'

/**
 * Sets up http handlers
 */
function startHTTPServer (
  app: Application, projectStore: ProjectStore, userManager: UserManager) {
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
 * Main function for backend server
 */
async function main (): Promise<void> {
  // initialize environment variables
  const env = makeEnv()

  // initialize storage and redis
  const storage = await makeStorage(env.database, env.data)
  const projectStore = new ProjectStore(env, storage)
  const userManager = new UserManager(storage)

  // start http and socket io servers
  const app: Application = express()
  const httpServer = createServer(app)
  const io = socketio(httpServer)

  // set up http handlers
  startHTTPServer(app, projectStore, userManager)

  // set up socket.io handler
  startSocketServer(io, env, projectStore, userManager)

  httpServer.listen(env.port)

  return
}

// TODO: Verify this is good promise handling
main().then().catch()
