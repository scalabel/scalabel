import {
  NextFunction,
  Request,
  Response
} from 'express'
import { Fields, Files } from 'formidable'
import { sprintf } from 'sprintf-js'
import {
  createProject, createTasks, parseFiles,
  parseForm, saveProject, saveTasks } from './create_project'
import * as types from './types'
import { getExistingProjects, logError } from './util'

/**
 * Logs requests to static or dynamic files
 */
export function LoggingHandler (
  req: Request, _res: Response, next: NextFunction) {
  const log = sprintf('Requesting %s', req.originalUrl)
  // tslint:disable-next-line
  console.info(log)
  next()
}

/**
 * Handles getting all projects' names
 */
export async function ProjectNameHandler (_req: Request, res: Response) {
  let projects: string[]
  const defaultProjects = ['No existing project']
  try {
    projects = await getExistingProjects()
    if (projects.length === 0) {
      projects = defaultProjects
    }
  } catch (err) {
    logError(err)
    projects = defaultProjects
  }
  const projectNames = JSON.stringify(projects)
  res.send(projectNames)
  res.end()
}

/**
 * Handles posted project form data
 */
export async function PostProjectHandler (req: Request, res: Response) {
  if (req.method !== 'POST' || req.fields === undefined) {
    res.sendStatus(404)
  }
  const fields = req.fields as Fields
  const files = req.files as Files
  try {
    // parse form from reqeust
    const form = await parseForm(fields)
    // parse item, category, and attribute data from the form
    const formFileData = await parseFiles(form.labelType, files)
    // create the project from the form data
    const project = await createProject(form, formFileData)
    await Promise.all([
      // save the project
      saveProject(project),
      // create tasks then save them
      createTasks(project).then(
        (tasks: types.Task[]) => saveTasks(tasks))
    ])
  } catch (err) {
    logError(err)
    // alert the user that something failed
    res.send(err.message)
  }
  res.end()
}
