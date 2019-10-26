import { MuiThemeProvider } from '@material-ui/core/styles'
import { cleanup, fireEvent, render, waitForElement } from '@testing-library/react'
import { createCanvas } from 'canvas'
import * as child from 'child_process'
import _ from 'lodash'
import * as React from 'react'
import Session, { ConnectionStatus } from '../../js/common/session'
import { submissionTimeout } from '../../js/components/create_form'
import TitleBar, { saveTimeout } from '../../js/components/title_bar'
import { Label2DList } from '../../js/drawable/2d/label2d_list'
import { getShape } from '../../js/functional/state_util'
import { RectType } from '../../js/functional/types'
import { Size2D } from '../../js/math/size2d'
import { Vector2D } from '../../js/math/vector2d'
import { myTheme } from '../../js/styles/theme'
import {
  changeTestConfig,
  deepDeleteTimestamp,
  deleteTestDir,
  getExport,
  getExportFromDisc,
  projectInitSession,
  sleep,
  StyledIntegrationForm,
  testConfig,
  waitForSave
} from './util'

// TODO: add testing with the actual canvas

let launchProc: child.ChildProcessWithoutNullStreams

beforeAll(async () => {
  Session.devMode = false
  Session.testMode = true
  launchProc = child.spawn('node', [
    'app/dist/js/main.js',
    '--config',
    './app/config/test_config.yml'
  ])
  /*launchProc.stdout.on('data', (data) => {
    process.stdout.write(data)
  })

  launchProc.stderr.on('data', (data) => {
    process.stdout.write(data)
  })*/
  window.alert = (): void => {
    return
  }
  // Needed as buffer period for theserver to launch. The amount of time needed
  // is inconsistent so this is on the convservative side.
  await sleep(2000)
})
beforeEach(() => {
  cleanup()
})
afterEach(cleanup)
afterAll(() => {
  launchProc.kill()
  deleteTestDir()
})

describe('full 2d bounding box integration test', () => {
  test('project creation', async () => {
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <StyledIntegrationForm />
      </MuiThemeProvider>
    )
    // change project meta-data
    const projectNameInput = getByTestId('project-name') as HTMLInputElement
    fireEvent.change(projectNameInput, { target: { value:
      testConfig.projectName } })
    expect(projectNameInput.value).toBe(testConfig.projectName)
    const itemSelect = getByTestId('item-type') as HTMLSelectElement
    fireEvent.change(itemSelect, { target: { value: 'image' } })
    expect(itemSelect.value).toBe('image')
    const labelSelect = getByTestId('label-type') as HTMLSelectElement
    fireEvent.change(labelSelect, { target: { value: 'box2d' } })
    expect(labelSelect.value).toBe('box2d')
    const tasksizeInput = getByTestId('tasksize-input') as HTMLInputElement
    fireEvent.change(tasksizeInput, { target: { value: '5' } })
    expect(tasksizeInput.value).toBe('5')
    // submit the project
    const submitButton = getByTestId('submit-button')
    fireEvent.click(submitButton)
    await Promise.race(
      [waitForElement(() => getByTestId('hidden-buttons')),
        sleep(submissionTimeout)
      ]
    )
  })

  test('test 2d-bounding-box annotation and save to disc', async () => {
    // Spawn a canvas and draw labels on this canvas
    // Uses similar code to drawable tests
    const synchronizer = await projectInitSession()
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <TitleBar
          title={'title'}
          instructionLink={'instructionLink'}
          dashboardLink={'dashboardLink'}
          autosave = {Session.autosave}
          synchronizer = {synchronizer}
        />
      </MuiThemeProvider>
    )
    const saveButton = getByTestId('Save')
    const labelCanvas = createCanvas(200, 200)
    const labelContext = labelCanvas.getContext('2d')
    const controlCanvas = createCanvas(200, 200)
    const controlContext = controlCanvas.getContext('2d')
    const handleIndex = 0
    const _labelIndex = -2
    let labelId = -1
    let state = Session.getState()
    const itemIndex = state.user.select.item
    const label2dList = new Label2DList()
    Session.subscribe(() => {
      label2dList.updateState(
        Session.getState(),
        Session.getState().user.select.item
      )
    })

    const canvasSize = new Size2D(100, 100)
    label2dList.onMouseDown(new Vector2D(1, 1), _labelIndex, handleIndex)
    for (let i = 1; i <= 10; i += 1) {
      label2dList.onMouseMove(new Vector2D(i, i), canvasSize, _labelIndex,
      handleIndex)
      label2dList.redraw(labelContext, controlContext, 1)
    }
    label2dList.onMouseUp(new Vector2D(10, 10), _labelIndex, handleIndex)
    label2dList.redraw(labelContext, controlContext, 1)
    labelId += 1

    state = Session.getState()
    expect(_.size(state.task.items[itemIndex].labels)).toEqual(1)
    let rect = getShape(state, itemIndex, labelId, 0) as RectType
    expect(rect.x1).toEqual(1)
    expect(rect.y1).toEqual(1)
    expect(rect.x2).toEqual(10)
    expect(rect.y2).toEqual(10)

    label2dList.onMouseDown(new Vector2D(20, 20), _labelIndex, handleIndex)
    for (let i = 20; i <= 40; i += 1) {
      label2dList.onMouseMove(new Vector2D(i, i), canvasSize, _labelIndex,
      handleIndex)
      label2dList.redraw(labelContext, controlContext, 1)
    }
    label2dList.onMouseUp(new Vector2D(40, 40), _labelIndex, handleIndex)
    label2dList.redraw(labelContext, controlContext, 1)
    labelId += 1

    state = Session.getState()
    expect(_.size(state.task.items[itemIndex].labels)).toEqual(2)
    rect = getShape(state, itemIndex, labelId, 0) as RectType
    expect(rect.x1).toEqual(20)
    expect(rect.y1).toEqual(20)
    expect(rect.x2).toEqual(40)
    expect(rect.y2).toEqual(40)
    // save to disc
    fireEvent.click(saveButton)
    await Promise.race([
      sleep(saveTimeout),
      waitForSave()
    ])
    expect(Session.status).toBe(ConnectionStatus.SAVED)
  }, saveTimeout)

  test('test export of saved bounding boxes', async () => {
    const exportJson = await getExport()
    const trueExportJson = getExportFromDisc()
    const noTimestampExportJson = deepDeleteTimestamp(exportJson)
    const noTimestampTrueExportJSon = deepDeleteTimestamp(trueExportJson)
    expect(noTimestampExportJson).toEqual(noTimestampTrueExportJSon)
    changeTestConfig({
      exportMode: true
    })
  })

  test('import exported json from saved bounding boxes', async () => {
    const { getByTestId } = render(
      <MuiThemeProvider theme={myTheme}>
        <StyledIntegrationForm />
      </MuiThemeProvider>
    )
    // change project meta-data
    const projectNameInput = getByTestId('project-name') as HTMLInputElement
    fireEvent.change(projectNameInput, { target: { value:
      testConfig.projectName + '_exported' } })
    expect(projectNameInput.value).toBe(testConfig.projectName + '_exported')
    const itemSelect = getByTestId('item-type') as HTMLSelectElement
    fireEvent.change(itemSelect, { target: { value: 'image' } })
    expect(itemSelect.value).toBe('image')
    const labelSelect = getByTestId('label-type') as HTMLSelectElement
    fireEvent.change(labelSelect, { target: { value: 'box2d' } })
    expect(labelSelect.value).toBe('box2d')
    const tasksizeInput = getByTestId('tasksize-input') as HTMLInputElement
    fireEvent.change(tasksizeInput, { target: { value: '5' } })
    expect(tasksizeInput.value).toBe('5')
    // submit the project
    const submitButton = getByTestId('submit-button')
    fireEvent.click(submitButton)
    await Promise.race(
      [waitForElement(() => getByTestId('hidden-buttons')),
        sleep(submissionTimeout)
      ])
  })
})
