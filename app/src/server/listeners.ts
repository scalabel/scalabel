import { NextFunction, Request, Response } from "express"
import { File } from "formidable"
import _ from "lodash"

import { DashboardContents } from "../components/dashboard"
import { getSubmissionTime } from "../components/util"
import { FormField } from "../const/project"
import { ItemExport } from "../types/export"
import { Project } from "../types/project"
import { TaskType } from "../types/state"
import {
  createProject,
  createTasks,
  parseFiles,
  parseForm,
  readConfig
} from "./create_project"
import { convertStateToExport } from "./export"
import { FileStorage } from "./file_storage"
import Logger from "./logger"
import { getExportName } from "./path"
import { ProjectStore } from "./project_store"
import { S3Storage } from "./s3_storage"
import { getProjectOptions, getProjectStats, getTaskOptions } from "./stats"
import { Storage } from "./storage"
import { UserManager } from "./user_manager"
import { parseProjectName } from "./util"

/**
 * Wraps HTTP listeners
 */
export class Listeners {
  /** the project store */
  protected projectStore: ProjectStore
  /** the user manager */
  protected userManager: UserManager

  /**
   * Constructor
   *
   * @param projectStore
   * @param userManager
   */
  constructor(projectStore: ProjectStore, userManager: UserManager) {
    this.projectStore = projectStore
    this.userManager = userManager
  }

  /**
   * Logs requests to static or dynamic files
   *
   * @param req
   * @param _res
   * @param next
   */
  public loggingHandler(
    req: Request,
    _res: Response,
    next: NextFunction
  ): void {
    const log = `Requesting ${req.originalUrl}`
    Logger.info(log)
    next()
  }

  /**
   * Handles getting all projects' names
   *
   * @param _req
   * @param res
   */
  public async projectNameHandler(_req: Request, res: Response): Promise<void> {
    let projects: string[]
    const defaultProjects = ["No existing project"]
    try {
      projects = await this.projectStore.getExistingProjects()
      if (projects.length === 0) {
        projects = defaultProjects
      }
    } catch (err) {
      Logger.error(err)
      projects = defaultProjects
    }
    const projectNames = JSON.stringify(projects)
    res.send(projectNames)
    res.end()
  }

  /**
   * Handles posting export
   *
   * @param req
   * @param res
   */
  public async getExportHandler(req: Request, res: Response): Promise<void> {
    if (this.checkInvalidGet(req, res)) {
      return
    }

    try {
      const projectName = req.query[FormField.PROJECT_NAME] as string
      // Grab the latest submissions from all tasks
      const tasks = await this.projectStore.getTasksInProject(projectName)
      let items: ItemExport[] = []
      // Load the latest submission for each task to export
      for (const task of tasks) {
        try {
          const taskId = task.config.taskId
          const state = await this.projectStore.loadState(projectName, taskId)
          items = items.concat(convertStateToExport(state))
        } catch (error) {
          Logger.info(error.message)
          for (const itemToLoad of task.items) {
            const url = Object.values(itemToLoad.urls)[0]
            const timestamp = getSubmissionTime(task.progress.submissions)
            items.push({
              name: url,
              url,
              sensor: -1,
              timestamp,
              videoName: itemToLoad.videoName,
              attributes: {},
              labels: []
            })
          }
        }
      }
      const exportJson = JSON.stringify(items, null, "  ")
      // Set relevant header and send the exported json file
      res.attachment(getExportName(projectName))
      res.end(Buffer.from(exportJson, "binary"), "binary")
    } catch (error) {
      // TODO: Be more specific about what this error may be
      Logger.error(error)
      res.end()
    }
  }

  /**
   * Alert the user that the sent fields were illegal
   *
   * @param res
   */
  public badFormResponse(res: Response): void {
    const err = Error("Illegal fields for project creation")
    Logger.error(err)
    res.status(400).send(err.message)
  }

  /**
   * Alert the user that the task creation request was illegal
   *
   * @param res
   */
  public badTaskResponse(res: Response): void {
    const err = Error("Illegal fields for task creation")
    Logger.error(err)
    res.status(400).send(err.message)
  }

  /**
   * Error if it's not a post request
   *
   * @param req
   * @param res
   */
  public checkInvalidPost(req: Request, res: Response): boolean {
    if (req.method !== "POST") {
      res.sendStatus(404)
      res.end()
      return true
    }
    return false
  }

  /**
   * Error if it's not a get request
   * By default, also requires non-empty queryArg parameters
   *
   * @param req
   * @param res
   * @param requireParam
   */
  public checkInvalidGet(
    req: Request,
    res: Response,
    requireParam: boolean = true
  ): boolean {
    if (req.method !== "GET" || (requireParam && req.query === {})) {
      res.sendStatus(404)
      res.end()
      return true
    }
    return false
  }

  /**
   * Handles posted project from internal data
   * Items file not required, since items can be added later
   *
   * @param req
   * @param res
   */
  public async postProjectInternalHandler(
    req: Request,
    res: Response
  ): Promise<void> {
    if (this.checkInvalidPost(req, res)) {
      return
    }

    if (
      req.body === undefined ||
      req.body.fields === undefined ||
      req.body.files === undefined
    ) {
      return this.badFormResponse(res)
    }

    /**
     * Use the region/bucket specified in the request
     * to access the item/category/attribute files
     */
    const s3Path = req.body.fields.s3_path as string
    let storage: Storage
    try {
      storage = new S3Storage(s3Path)
    } catch (err) {
      Logger.error(err)
      return this.badFormResponse(res)
    }
    storage.setExt("")
    await this.createProjectFromDicts(
      storage,
      req.body.fields,
      req.body.files,
      false,
      res
    )
  }

  /**
   * Handles posted project from form data
   * Items file required
   *
   * @param req
   * @param res
   */
  public async postProjectHandler(req: Request, res: Response): Promise<void> {
    if (this.checkInvalidPost(req, res)) {
      return
    }

    if (req.fields === undefined || req.files === undefined) {
      return this.badFormResponse(res)
    }

    const fields = req.fields as { [key: string]: string }
    const formFiles = req.files as { [key: string]: File | undefined }
    const files: { [key: string]: string } = {}
    for (const key of Object.keys(formFiles)) {
      const file = formFiles[key]
      if (file !== undefined && file.size !== 0) {
        files[key] = file.path
      }
    }

    const storage = new FileStorage("")
    storage.setExt("")
    await this.createProjectFromDicts(storage, fields, files, true, res)
  }

  /**
   * Handles tasks being added to a project
   *
   * @param req
   * @param res
   */
  public async postTasksHandler(req: Request, res: Response): Promise<void> {
    if (this.checkInvalidPost(req, res)) {
      return
    }

    if (
      req.body === undefined ||
      req.body.items === undefined ||
      req.body.projectName === undefined
    ) {
      this.badTaskResponse(res)
      return
    }

    // Read in the data
    const storage = new FileStorage("")
    storage.setExt("")
    const items = await readConfig<Array<Partial<ItemExport>>>(
      storage,
      req.body.items,
      []
    )
    let project: Project
    let projectName: string
    try {
      projectName = parseProjectName(req.body.projectName)
      project = await this.projectStore.loadProject(projectName)
    } catch (err) {
      Logger.error(err)
      this.badTaskResponse(res)
      return
    }

    // Update the project with the new items
    const itemStartNum = project.items.length
    project.items = project.items.concat(items)
    await this.projectStore.saveProject(project)

    // Update the tasks, make sure not to combine old and new items
    project.items = items
    const oldTasks = await this.projectStore.getTasksInProject(projectName)
    const taskStartNum = oldTasks.length
    const tasks = await createTasks(project, taskStartNum, itemStartNum)
    await this.projectStore.saveTasks(tasks)

    res.sendStatus(200)
  }

  /**
   * Get the labeling stats
   *
   * @param req
   * @param res
   */
  public async statsHandler(req: Request, res: Response): Promise<void> {
    if (this.checkInvalidGet(req, res)) {
      return
    }

    try {
      const projectName = req.query.name as string
      const savedTasks = await this.projectStore.loadTaskStates(projectName)
      const stats = getProjectStats(savedTasks)
      res.send(JSON.stringify(stats))
    } catch (err) {
      Logger.error(err)
      res.send(err.message)
    }
  }

  /**
   * Return dashboard info
   *
   * @param req
   * @param res
   */
  public async dashboardHandler(req: Request, res: Response): Promise<void> {
    if (this.checkInvalidGet(req, res)) {
      return
    }

    try {
      const projectName = req.query.name as string

      const project = await this.projectStore.loadProject(projectName)
      const projectMetaData = getProjectOptions(project)

      const savedTasks = await this.projectStore.loadTaskStates(projectName)
      const taskOptions = _.map(savedTasks, getTaskOptions)

      const numUsers = await this.userManager.countUsers(projectName)
      const contents: DashboardContents = {
        projectMetaData,
        taskMetaDatas: taskOptions,
        numUsers
      }

      res.send(JSON.stringify(contents))
    } catch (err) {
      Logger.error(err)
      res.send(err.message)
    }
  }

  /**
   * Finishes project creation using processed dicts
   *
   * @param storage
   * @param itemsRequired
   * @param res
   */
  private async createProjectFromDicts(
    storage: Storage,
    fields: { [key: string]: string },
    files: { [key: string]: string },
    itemsRequired: boolean,
    res: Response
  ): Promise<void> {
    try {
      // Parse form from request
      const form = await parseForm(fields, this.projectStore)
      // Parse item, category, and attribute data from the form
      const formFileData = await parseFiles(
        storage,
        form.labelType,
        files,
        itemsRequired
      )
      // Create the project from the form data
      const project = await createProject(form, formFileData)
      await Promise.all([
        this.projectStore.saveProject(project),
        // Create tasks then save them
        createTasks(project).then(
          async (tasks: TaskType[]) => await this.projectStore.saveTasks(tasks)
        )
        // Save the project
      ])
      res.send()
    } catch (err) {
      Logger.error(err)
      // Alert the user that something failed
      res.status(400).send(err.message)
    }
  }
}
