import { ListItemText } from "@material-ui/core"
import ToggleButton from "@material-ui/lab/ToggleButton"
import { cleanup, fireEvent, render } from "@testing-library/react"
import _ from "lodash"
import * as React from "react"
import { create } from "react-test-renderer"

import * as action from "../../src/action/common"
import { selectLabel } from "../../src/action/select"
import Session, { dispatch, getState } from "../../src/common/session"
import { ToolBar } from "../../src/components/toolbar"
import { Category } from "../../src/components/toolbar_category"
import { ListButton } from "../../src/components/toolbar_list_button"
import { makeLabel } from "../../src/functional/states"
import { testJson } from "../test_states/test_image_objects"
import { testJson as testTrackJson } from "../test_states/test_track_objects"
import { setupTestStore } from "./util"

let handleToggleWasCalled: boolean = false
const testValues = ["NA", "A", "B", "C"]
const selected: { [key: string]: string } = {}

/**
 * dummy attribute toggle function to test if the correct toggle action was
 * correctly called
 *
 * @param toggleName
 * @param alignment
 * @param isTag
 */
const dummyAttributeToggle = (toggleName: string, alignment: string): void => {
  selected[toggleName] = alignment
  handleToggleWasCalled = true
}

const dummyGetAlignmentIndex = (toggleName: string): number => {
  const index = testValues.indexOf(selected[toggleName])
  return index >= 0 ? index : 0
}

beforeEach(() => {
  handleToggleWasCalled = false
})

afterEach(cleanup)

describe("Toolbar category setting", () => {
  test("Category selection", () => {
    setupTestStore(testJson)
    const { getByText } = render(
      <Category categories={["A", "B", "C"]} headerText={"Category"} />
    )
    let button = getByText("B")
    expect(button).toBeInTheDocument()
    fireEvent.click(button)
    expect(getState().user.select.category).toBe(1)
    button = getByText("C")
    expect(button).toBeInTheDocument()
    fireEvent.click(button)
    expect(getState().user.select.category).toBe(2)
  })

  test("Test elements in Category", () => {
    const category = create(
      <Category categories={["A", "B"]} headerText={"Category"} />
    )
    const root = category.root
    expect(root.props.categories[0].toString()).toBe("A")
    expect(root.props.categories[1].toString()).toBe("B")
    expect(root.findByType(ListItemText).props.primary).toBe("Category")
  })

  test("Category by type", () => {
    const category = create(
      <Category categories={["OnlyCategory"]} headerText={"Category"} />
    )
    const root = category.root
    expect(root.findByType(ToggleButton).props.children).toBe("OnlyCategory")
  })

  test("Null category", () => {
    const category = create(
      <Category categories={null} headerText={"Category"} />
    )
    const root = category.getInstance()
    expect(root).toBe(null)
  })
})

describe("test Delete", () => {
  test("Delete by keyboard", () => {
    setupTestStore(testJson)

    const toolbarRef: React.Ref<ToolBar> = React.createRef()
    render(
      <ToolBar
        ref={toolbarRef}
        categories={null}
        attributes={[]}
        labelType={"labelType"}
      />
    )
    expect(toolbarRef.current).not.toBeNull()
    expect(toolbarRef.current).not.toBeUndefined()
    if (toolbarRef.current !== null) {
      toolbarRef.current.componentDidMount()
    }
    for (let itemIndex = 0; itemIndex < 3; itemIndex += 1) {
      dispatch(action.goToItem(itemIndex))
      dispatch(action.addLabel(itemIndex, makeLabel()))
      const label = makeLabel()
      dispatch(action.addLabel(itemIndex, label))
      dispatch(action.addLabel(itemIndex, makeLabel()))
      dispatch(action.addLabel(itemIndex, makeLabel()))
      fireEvent.keyDown(document, { key: "Backspace" })
      let item = getState().task.items[itemIndex]
      expect(_.size(item.labels)).toBe(3)
      expect(label.id in item.labels).toBe(true)
      dispatch(selectLabel(getState().user.select.labels, itemIndex, label.id))
      fireEvent.keyDown(document, { key: "Backspace" })
      item = getState().task.items[itemIndex]
      expect(_.size(item.labels)).toBe(2)
      expect(label.id in item.labels).toBe(false)
    }
  })
})

describe("test functionality for attributes with multiple values", () => {
  test("proper initialization", () => {
    const { getByTestId } = render(
      <ListButton
        name={"test"}
        values={testValues}
        handleAttributeToggle={dummyAttributeToggle}
        getAlignmentIndex={dummyGetAlignmentIndex}
      />
    )
    expect(handleToggleWasCalled).toBe(false)
    const NAButton = getByTestId("toggle-button-NA") as HTMLButtonElement
    expect(NAButton.className).toContain("selected")
  })
  test("changing selected attribute calls callback for tag labeling", () => {
    const { getByTestId } = render(
      <ListButton
        name={"test"}
        values={testValues}
        handleAttributeToggle={dummyAttributeToggle}
        getAlignmentIndex={dummyGetAlignmentIndex}
      />
    )
    let AButton = getByTestId("toggle-button-A") as HTMLButtonElement
    expect(handleToggleWasCalled).toBe(false)
    fireEvent.click(AButton)
    AButton = getByTestId("toggle-button-A") as HTMLButtonElement
    let NAButton = getByTestId("toggle-button-NA") as HTMLButtonElement
    expect(handleToggleWasCalled).toBe(true)
    AButton = getByTestId("toggle-button-A") as HTMLButtonElement
    expect(AButton.className).toContain("selected")
    expect(NAButton.className).not.toContain("selected")
    handleToggleWasCalled = false
    fireEvent.click(NAButton)
    NAButton = getByTestId("toggle-button-NA") as HTMLButtonElement
    expect(handleToggleWasCalled).toBe(true)
    expect(NAButton.className).toContain("selected")
  })
})

describe("test track", () => {
  test("Delete by click toolbar button", () => {
    setupTestStore(testTrackJson)
    Session.images.length = 0
    Session.images.push({ [-1]: new Image(1000, 1000) })
    for (let i = 0; i < getState().task.items.length; i++) {
      dispatch(action.loadItem(i, -1))
    }

    const toolbarRef: React.Ref<ToolBar> = React.createRef()
    const { getByText } = render(
      <ToolBar
        ref={toolbarRef}
        categories={null}
        attributes={[]}
        labelType={"labelType"}
      />
    )
    expect(toolbarRef.current).not.toBeNull()
    expect(toolbarRef.current).not.toBeUndefined()
    if (toolbarRef.current !== null) {
      toolbarRef.current.componentDidMount()
    }

    dispatch(action.goToItem(1))
    let state = getState()
    const trackLabels = state.task.tracks[3].labels
    const lblInItm2 = trackLabels[2]
    const lblInItm3 = trackLabels[3]
    const lblInItm4 = trackLabels[4]
    const lblInItm5 = trackLabels[5]
    expect(_.size(state.task.tracks[3].labels)).toBe(6)
    expect(_.size(state.task.items[2].labels)).toBe(3)
    expect(_.size(state.task.items[2].shapes)).toBe(3)
    dispatch(selectLabel(getState().user.select.labels, 1, trackLabels[1]))
    fireEvent(
      getByText("Delete"),
      new MouseEvent("click", {
        bubbles: true,
        cancelable: true
      })
    )
    state = getState()
    expect(_.size(state.task.tracks[3].labels)).toBe(1)
    expect(state.task.items[2].labels[lblInItm2]).toBeUndefined()
    expect(state.task.items[2].labels[lblInItm3]).toBeUndefined()
    expect(state.task.items[2].labels[lblInItm4]).toBeUndefined()
    expect(state.task.items[2].labels[lblInItm5]).toBeUndefined()
  })
  test("Merge by click toolbar button", () => {
    setupTestStore(testTrackJson)

    const toolbarRef: React.Ref<ToolBar> = React.createRef()
    const { getByText } = render(
      <ToolBar
        ref={toolbarRef}
        categories={null}
        attributes={[]}
        labelType={"labelType"}
      />
    )
    expect(toolbarRef.current).not.toBeNull()
    expect(toolbarRef.current).not.toBeUndefined()
    if (toolbarRef.current !== null) {
      toolbarRef.current.componentDidMount()
    }

    dispatch(action.goToItem(3))
    let state = getState()
    expect(_.size(state.task.tracks[2].labels)).toBe(4)
    expect(_.size(state.task.tracks[9].labels)).toBe(1)
    expect(state.task.items[5].labels["203"].track).toEqual("9")

    dispatch(selectLabel(getState().user.select.labels, 3, "49"))
    fireEvent(
      getByText("Link Tracks"),
      new MouseEvent("click", {
        bubbles: true,
        cancelable: true
      })
    )

    dispatch(action.goToItem(5))
    dispatch(selectLabel(getState().user.select.labels, 5, "203"))
    fireEvent(
      getByText("Finish"),
      new MouseEvent("click", {
        bubbles: true,
        cancelable: true
      })
    )

    state = getState()
    expect(_.size(state.task.tracks["2"].labels)).toBe(5)
    expect(state.task.tracks["9"]).toBeUndefined()
    expect(state.task.items[5].labels["203"].track).toEqual("2")
  })
})
