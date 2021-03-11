import { ThemeProvider } from "@material-ui/core/styles"
import {
  cleanup,
  fireEvent,
  render,
  waitForElement
} from "@testing-library/react"
import axios from "axios"
import { createCanvas } from "canvas"
import * as child from "child_process"
import _ from "lodash"
import * as React from "react"
import { Provider } from "react-redux"

import Session from "../../src/common/session"
import { initSessionForTask } from "../../src/common/session_init"
import { submissionTimeout } from "../../src/components/create_form"
import TitleBar, { saveTimeout } from "../../src/components/title_bar"
import { Endpoint } from "../../src/const/connection"
import { Label2DHandler } from "../../src/drawable/2d/label2d_handler"
import { Label2DList } from "../../src/drawable/2d/label2d_list"
import { isStatusSaved } from "../../src/functional/selector"
import { getShape } from "../../src/functional/state_util"
import { Size2D } from "../../src/math/size2d"
import { Vector2D } from "../../src/math/vector2d"
import { scalabelTheme } from "../../src/styles/theme"
import { IdType, RectType } from "../../src/types/state"
import { getTestConfig, getTestConfigPath } from "../server/util/util"
import { findNewLabelsFromState } from "../util/state"
import {
  changeTestConfig,
  countTasks,
  deepDeleteTimestamp,
  deleteTestDir,
  getExport,
  getExportFromDisc,
  getProjectJson,
  getProjectJsonFromDisc,
  sleep,
  StyledIntegrationForm,
  testConfig,
  waitForSave
} from "./util"

// TODO: add testing with the actual canvas

let launchProc: child.ChildProcessWithoutNullStreams

beforeAll(async () => {
  Session.testMode = true

  // Port is also changed in test_config
  launchProc = child.spawn("node", [
    "app/dist/main.js",
    "--config",
    getTestConfigPath()
  ])

  // LaunchProc.stdout.on('data', (data) => {
  //   process.stdout.write(data)
  // })

  // LaunchProc.stderr.on('data', (data) => {
  //   process.stdout.write(data)
  // })

  window.alert = (): void => {}
  // Needed as buffer period for server to launch. The amount of time needed
  // is inconsistent so this is on the convservative side.
  await sleep(1500)
})
beforeEach(() => {
  cleanup()
})
afterEach(cleanup)
afterAll(async () => {
  launchProc.kill()
  deleteTestDir()
  // Wait for server to shut down to clear port
  await sleep(50)
})

test("project creation", async () => {
  const { getByTestId } = render(
    <ThemeProvider theme={scalabelTheme}>
      <StyledIntegrationForm />
    </ThemeProvider>
  )
  // Change project meta-data
  const projectNameInput = getByTestId("project-name") as HTMLInputElement
  fireEvent.change(projectNameInput, {
    target: { value: testConfig.projectName }
  })
  expect(projectNameInput.value).toBe(testConfig.projectName)
  const itemSelect = getByTestId("item-type") as HTMLSelectElement
  fireEvent.change(itemSelect, { target: { value: "image" } })
  expect(itemSelect.value).toBe("image")
  const labelSelect = getByTestId("label-type") as HTMLSelectElement
  fireEvent.change(labelSelect, { target: { value: "box2d" } })
  expect(labelSelect.value).toBe("box2d")
  const tasksizeInput = getByTestId("tasksize-input") as HTMLInputElement
  fireEvent.change(tasksizeInput, { target: { value: "5" } })
  expect(tasksizeInput.value).toBe("5")
  // Submit the project
  const submitButton = getByTestId("submit-button")
  fireEvent.click(submitButton)
  await Promise.race([
    waitForElement(() => getByTestId("hidden-buttons")),
    sleep(submissionTimeout)
  ])
})
test("test project.json was properly created", () => {
  const projectJson = getProjectJson()
  const sampleProjectJson = getProjectJsonFromDisc()
  expect(projectJson).toEqual(sampleProjectJson)
})

test(
  "test 2d-bounding-box annotation and save to disc",
  async () => {
    // Spawn a canvas and draw labels on this canvas
    // Uses similar code to drawable tests
    initSessionForTask(
      testConfig.taskIndex,
      testConfig.projectName,
      "fakeId",
      "",
      false
    )

    const labelIds: IdType[] = []
    const { getByTestId } = render(
      <ThemeProvider theme={scalabelTheme}>
        <Provider store={Session.store}>
          <TitleBar />
        </Provider>
      </ThemeProvider>
    )
    const saveButton = getByTestId("Save")
    const labelCanvas = createCanvas(200, 200)
    const labelContext = labelCanvas.getContext("2d")
    const controlCanvas = createCanvas(200, 200)
    const controlContext = controlCanvas.getContext("2d")
    const handleIndex = 0
    const _labelIndex = -2
    let state = Session.getState()
    const itemIndex = state.user.select.item
    const label2dList = new Label2DList()
    const label2dHandler = new Label2DHandler(Session.label2dList)
    Session.subscribe(() => {
      const newState = Session.getState()
      Session.label2dList.updateState(newState)
      label2dHandler.updateState(newState)
    })

    const canvasSize = new Size2D(100, 100)
    label2dHandler.onMouseMove(
      new Vector2D(1, 1),
      canvasSize,
      _labelIndex,
      handleIndex
    )
    label2dHandler.onMouseDown(new Vector2D(1, 1), _labelIndex, handleIndex)
    for (let i = 1; i <= 10; i += 1) {
      label2dHandler.onMouseMove(
        new Vector2D(i, i),
        canvasSize,
        _labelIndex,
        handleIndex
      )
      label2dList.redraw(labelContext, controlContext, 1)
    }
    label2dHandler.onMouseUp(new Vector2D(10, 10), _labelIndex, handleIndex)
    label2dList.redraw(labelContext, controlContext, 1)
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])

    state = Session.getState()
    expect(_.size(state.task.items[itemIndex].labels)).toEqual(1)
    let rect = getShape(state, itemIndex, labelIds[0], 0) as RectType
    expect(rect.x1).toEqual(1)
    expect(rect.y1).toEqual(1)
    expect(rect.x2).toEqual(10)
    expect(rect.y2).toEqual(10)

    label2dHandler.onMouseMove(
      new Vector2D(20, 20),
      canvasSize,
      _labelIndex,
      handleIndex
    )
    label2dHandler.onMouseDown(new Vector2D(20, 20), _labelIndex, handleIndex)
    for (let i = 20; i <= 40; i += 1) {
      label2dHandler.onMouseMove(
        new Vector2D(i, i),
        canvasSize,
        _labelIndex,
        handleIndex
      )
      label2dList.redraw(labelContext, controlContext, 1)
    }
    label2dHandler.onMouseUp(new Vector2D(40, 40), _labelIndex, handleIndex)
    label2dList.redraw(labelContext, controlContext, 1)

    state = Session.getState()
    expect(_.size(state.task.items[itemIndex].labels)).toEqual(2)
    labelIds.push(findNewLabelsFromState(Session.getState(), 0, labelIds)[0])
    rect = getShape(state, itemIndex, labelIds[1], 0) as RectType
    expect(rect.x1).toEqual(20)
    expect(rect.y1).toEqual(20)
    expect(rect.x2).toEqual(40)
    expect(rect.y2).toEqual(40)
    // Save to disc
    fireEvent.click(saveButton)
    await Promise.race([sleep(saveTimeout), waitForSave()])
    expect(isStatusSaved(Session.store.getState())).toBe(true)
  },
  saveTimeout
)

test("test export of saved bounding boxes", async () => {
  const exportJson = await getExport()
  const trueExportJson = getExportFromDisc()
  const noTimestampExportJson = deepDeleteTimestamp(exportJson)
  const noTimestampTrueExportJson = deepDeleteTimestamp(trueExportJson)
  expect(noTimestampExportJson).toEqual(noTimestampTrueExportJson)
  changeTestConfig({
    exportMode: true
  })
})

test("import exported json from saved bounding boxes", async () => {
  const { getByTestId } = render(
    <ThemeProvider theme={scalabelTheme}>
      <StyledIntegrationForm />
    </ThemeProvider>
  )
  // Change project meta-data
  const projectNameInput = getByTestId("project-name") as HTMLInputElement
  fireEvent.change(projectNameInput, {
    target: { value: testConfig.projectName + "_exported" }
  })
  expect(projectNameInput.value).toBe(testConfig.projectName + "_exported")
  const itemSelect = getByTestId("item-type") as HTMLSelectElement
  fireEvent.change(itemSelect, { target: { value: "image" } })
  expect(itemSelect.value).toBe("image")
  const labelSelect = getByTestId("label-type") as HTMLSelectElement
  fireEvent.change(labelSelect, { target: { value: "box2d" } })
  expect(labelSelect.value).toBe("box2d")
  const tasksizeInput = getByTestId("tasksize-input") as HTMLInputElement
  fireEvent.change(tasksizeInput, { target: { value: "5" } })
  expect(tasksizeInput.value).toBe("5")
  // Submit the project
  const submitButton = getByTestId("submit-button")
  fireEvent.click(submitButton)
  await Promise.race([
    waitForElement(() => getByTestId("hidden-buttons")),
    sleep(submissionTimeout)
  ])
})

describe("2d bounding box integration test with programmatic api", () => {
  const axiosConfig = {
    headers: {
      "Content-Type": "application/json"
    }
  }
  const serverConfig = getTestConfig()
  const response400 = "Request failed with status code 400"

  test("Itemless project creation", async () => {
    // Make task size big enough to fit all the items in 1 task
    const projectName = "itemless-project"
    const createProjectBody = {
      fields: {
        project_name: projectName,
        item_type: "image",
        label_type: "box2d",
        task_size: 400
      },
      files: {}
    }
    const address = new URL("http://localhost")
    address.port = serverConfig.http.port.toString()
    address.pathname = Endpoint.POST_PROJECT_INTERNAL
    const response = await axios.post(
      address.toString(),
      createProjectBody,
      axiosConfig
    )
    expect(response.status).toBe(200)
    expect(await countTasks(projectName)).toBe(0)

    // Trying to create a project with the same name fails
    await expect(
      axios.post(address.toString(), createProjectBody, axiosConfig)
    ).rejects.toThrow(response400)

    // Add items to the empty project; expect 1 task to be made
    address.pathname = Endpoint.POST_TASKS
    const addTasksBody = {
      projectName,
      items: "examples/image_list.yml"
    }
    await axios.post(address.toString(), addTasksBody, axiosConfig)
    expect(await countTasks(projectName)).toBe(1)

    /* Add items again, check that a new task is made
     * even though the items COULD all fit in one task     */
    await axios.post(address.toString(), addTasksBody, axiosConfig)
    expect(await countTasks(projectName)).toBe(2)
  })
})
