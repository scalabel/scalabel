import { NextFunction, Request, Response } from "express"
import { File } from "formidable"
import { filterXSS } from "xss"

import { DashboardContents, TaskOptions } from "../components/dashboard"
import { getSubmissionTime } from "../components/util"
import { FormField } from "../const/project"
import { DatasetExport, ItemExport } from "../types/export"
import { Project } from "../types/project"
import {
  createProject,
  createTasks,
  parseFiles,
  parseSingleFile,
  parseForm,
  readConfig,
  filterIntersectedPolygonsInProject
} from "./create_project"
import { convertStateToExport } from "./export"
import { FileStorage } from "./file_storage"
import Logger from "./logger"
import { getExportName } from "./path"
import { ProjectStore } from "./project_store"
import { S3Storage } from "./s3_storage"
import {
  getDefaultTaskOptions,
  getProjectOptions,
  getProjectStats,
  getTaskOptions
} from "./stats"
import { Storage } from "./storage"
import { UserManager } from "./user_manager"
import { parseProjectName } from "./util"
import { QueryArg } from "../const/common"

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
      Logger.error(err as Error)
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
      const dataset: DatasetExport = {
        frames: [],
        config: {
          attributes: [],
          categories: []
        }
      }
      let items: ItemExport[] = []
      // Load the latest submission for each task to export
      for (const task of tasks) {
        try {
          const taskId = task.config.taskId
          const state = await this.projectStore.loadState(projectName, taskId)
          items = items.concat(convertStateToExport(state))
          if (dataset.config.attributes?.length === 0) {
            dataset.config.attributes = state.task.config.attributes
          }
          if (dataset.config.categories?.length === 0) {
            dataset.config.categories = state.task.config.treeCategories
          }
        } catch (error) {
          Logger.info((error as Error).message)
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
      dataset.frames = items
      const exportJson = JSON.stringify(dataset, null, "  ")
      // Set relevant header and send the exported json file
      res.attachment(getExportName(projectName))
      res.end(Buffer.from(exportJson, "binary"), "binary")
    } catch (error) {
      // TODO: Be more specific about what this error may be
      Logger.error(error as Error)
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
      Logger.error(err as Error)
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
        files[key] = file.filepath
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

    // TODO: This if clause aims to solve the lgtm alert.
    // Could be removed in the future if better way found.
    if (req.body.items !== "examples/image_list.yml") {
      throw Error(`req.body.items should be "examples/image_list.yml" here.`)
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
      Logger.error(err as Error)
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
    await createTasks(project, this.projectStore, taskStartNum, itemStartNum)

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
      Logger.error(err as Error)
      res.send(filterXSS((err as Error).message))
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

      const project = await this.projectStore.loadProjectInfo(projectName)
      const projectMetaData = getProjectOptions(project)

      const taskKeys = await this.projectStore.getTaskKeysInProject(projectName)

      let taskOptions: TaskOptions[] = []
      taskKeys.forEach((_taskKey: string) => {
        taskOptions = taskOptions.concat(getDefaultTaskOptions(project.config))
      })

      const numUsers = await this.userManager.countUsers(projectName)
      const contents: DashboardContents = {
        projectMetaData,
        taskMetaDatas: taskOptions,
        taskKeys,
        numUsers
      }

      res.send(JSON.stringify(contents))
    } catch (err) {
      Logger.error(err as Error)
      res.send(filterXSS((err as Error).message))
    }
  }

  /**
   * Delete a project
   *
   * @param req
   * @param res
   */
  public async deleteProjectHandler(
    req: Request,
    res: Response
  ): Promise<void> {
    if (this.checkInvalidGet(req, res)) {
      return
    }

    await this.projectStore.deleteProject(
      req.query[FormField.PROJECT_NAME] as string
    )

    res.sendStatus(200)
  }

  /**
   * Return task metadata
   *
   * @param req
   * @param res
   */
  public async taskMetaDataHandler(req: Request, res: Response): Promise<void> {
    if (this.checkInvalidGet(req, res)) {
      return
    }

    try {
      const projectName = req.query[QueryArg.PROJECT_NAME] as string
      const taskId = req.query[QueryArg.TASK_ID] as string

      const savedState = await this.projectStore.loadState(projectName, taskId)
      const savedTask = savedState.task
      const taskOption = getTaskOptions(savedTask)

      res.send(JSON.stringify(taskOption))
    } catch (err) {
      Logger.error(err as Error)
      res.send(filterXSS((err as Error).message))
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
      const formFileData =
        Object.keys(files).length > 1
          ? await parseFiles(storage, form.labelType, files, itemsRequired)
          : await parseSingleFile(storage, form.labelType, files)
      // Create the project from the form data
      const project = await createProject(form, formFileData)
      const [filteredProject, msg] = filterIntersectedPolygonsInProject(project)

      await Promise.all([
        this.projectStore.saveProject(filteredProject),
        // Create tasks then save them
        createTasks(filteredProject, this.projectStore)
      ])
      res.send(filterXSS(msg))
    } catch (err) {
      Logger.error(err as Error)
      // Alert the user that something failed
      res.status(400).send(filterXSS((err as Error).message))
    }
  }
}
