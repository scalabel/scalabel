import "react-toastify/dist/ReactToastify.css"
import List from "@material-ui/core/List/List"
// import ListItem from "@material-ui/core/ListItem"
import _ from "lodash"
import React from "react"
import { Slide, toast, ToastContainer } from "react-toastify"

import {
  changeModeToAnnotating,
  changeModeToSelecting,
  changeOverlays,
  changeSelect,
  changeViewerConfig,
  mergeTracks,
  splitTrack,
  startLinkTrack
} from "../action/common"
import { activateSpan, deactivateSpan } from "../action/span3d"
import {
  changeSelectedLabelsAttributes,
  deleteSelectedLabels,
  deleteSelectedLabelsfromTracks,
  terminateSelectedTracks
} from "../action/select"
import { addLabelTag } from "../action/tag"
import { renderTemplate } from "../common/label"
import Session from "../common/session"
import { Key, LabelTypeName, ViewerConfigTypeName } from "../const/common"
import { getSelectedTracks } from "../functional/state_util"
import { isValidId, makeTrack } from "../functional/states"
import { tracksOverlapping } from "../functional/track"
import { Attribute, Category, ModeStatus, State } from "../types/state"
import { makeButton } from "./button"
import { Component } from "./component"
import { ToolbarCategory } from "./toolbar_category"
import { alert } from "../common/alert"
import { Severity } from "../types/common"

/** This is the interface of props passed to ToolBar */
interface Props {
  /** categories of ToolBar */
  categories: string[] | null
  /** treeCategories of Toolbar */
  treeCategories: Category[] | null
  /** attributes of ToolBar */
  attributes: Attribute[]
  /** labelType of ToolBar 'box2d' | 'polygon2d' | 'lane' */
  labelType: string
}
/**
 * This is ToolBar component that displays
 * all the attributes and categories for the 2D bounding box labeling tool
 */
export class ToolBar extends Component<Props> {
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** key down handler */
  private readonly _keyDownHandler: (e: KeyboardEvent) => void
  /** key up handler */
  private readonly _keyUpHandler: (e: KeyboardEvent) => void

  /**
   * Constructor
   *
   * @param props
   */
  constructor(props: Readonly<Props>) {
    super(props)
    this.handleToggle = this.handleToggle.bind(this)
    this._keyDownHandler = this.onKeyDown.bind(this)
    this._keyUpHandler = this.onKeyUp.bind(this)
    this.handleAttributeToggle = this.handleAttributeToggle.bind(this)
    this.getAlignmentIndex = this.getAlignmentIndex.bind(this)
    this._keyDownMap = {}
  }

  /**
   * handles keyDown Events
   *
   * @param {keyboardEvent} e
   */
  public onKeyDown(e: KeyboardEvent): void {
    switch (e.key) {
      case Key.BACKSPACE:
        this.deletePressed()
        break
      case Key.H_LOW:
      case Key.H_UP: {
        e.preventDefault()
        const config = {
          ...this.state.user.viewerConfigs[Session.activeViewerId]
        }
        config.hideLabels = !config.hideLabels
        Session.dispatch(changeViewerConfig(Session.activeViewerId, config))
        break
      }
      case Key.T_LOW:
      case Key.T_UP: {
        e.preventDefault()
        const config = {
          ...this.state.user.viewerConfigs[Session.activeViewerId]
        }
        config.hideTags = !config.hideTags
        Session.dispatch(changeViewerConfig(Session.activeViewerId, config))
        break
      }
      case Key.X_LOW: {
        if (this.state.session.mode === ModeStatus.ANNOTATING) {
          Session.dispatch(changeModeToSelecting())
          toast("Change to SELECTING mode.", {
            position: "top-center",
            autoClose: 2000
          })
        } else {
          Session.dispatch(changeModeToAnnotating())
          toast("Change to ANNOTATING mode.", {
            position: "top-center",
            autoClose: 2000
          })
        }
        break
      }
      case Key.ONE: {
        if (this.state.session.overlayStatus.includes(1)){
          let cur_state = [...this.state.session.overlayStatus]
          cur_state.splice(cur_state.indexOf(1),1)
          Session.dispatch(changeOverlays(cur_state))
        } else {
          let cur_state = [...this.state.session.overlayStatus]
          cur_state.push(1)
          Session.dispatch(changeOverlays(cur_state))
        }
        
        break
      }
      case Key.TWO:{
        if (this.state.session.overlayStatus.includes(2)){
          let cur_state = [...this.state.session.overlayStatus]
          cur_state.splice(cur_state.indexOf(2),1)
          Session.dispatch(changeOverlays(cur_state))
        } else {
          let cur_state = [...this.state.session.overlayStatus]
          cur_state.push(2)
          Session.dispatch(changeOverlays(cur_state))
        }
        break
      }
      // Currently the first switch attribute gets toggled on key press
      case Key.V_LOW: {
        if (this.props.attributes.length > 0) {
          if (this.props.attributes[0].type === "switch") {
            this.handleToggle(this.props.attributes[0].name)
          }
      
        }
        break
      } 
      
    }
    this._keyDownMap[e.key] = true
  }

  /**
   * Key up handler
   *
   * @param e
   */
  public onKeyUp(e: KeyboardEvent): void {
    // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
    delete this._keyDownMap[e.key]
  }

  /**
   * Whether a specific key is pressed down
   *
   * @param {string} key - the key to check
   * @return {boolean}
   */
  protected isKeyDown(key: string): boolean {
    return this._keyDownMap[key]
  }

  /**
   * Add keyDown Event Listener
   */
  public componentDidMount(): void {
    super.componentDidMount()
    document.addEventListener("keydown", this._keyDownHandler)
    document.addEventListener("keyup", this._keyUpHandler)
  }

  /**
   * Remove keyDown Event Listener
   */
  public componentWillUnmount(): void {
    super.componentWillUnmount()
    document.removeEventListener("keydown", this._keyDownHandler)
    document.removeEventListener("keyup", this._keyUpHandler)
  }

  /**
   * ToolBar render function
   *
   * @return component
   */
  public render(): React.ReactNode {
    const { categories, treeCategories, attributes } = this.props
    return (
      <div>
        {categories !== null ? (
          <ToolbarCategory
            categories={categories}
            treeCategories={treeCategories}
            headerText={"Category"}
          />
        ) : null}
        <List>
          {attributes.map((element: Attribute) => (
            <React.Fragment key={element.name}>
              {renderTemplate(
                element.type,
                this.handleToggle,
                this.handleAttributeToggle,
                this.getAlignmentIndex,
                element.name,
                element.values
              )}
            </React.Fragment>
          ))}
        </List>
        <div>
          <div>
            {makeButton("Delete", () => {
              this.deletePressed()
            })}
          </div>
          {(this.state.user.viewerConfigs[0].type ===
            ViewerConfigTypeName.POINT_CLOUD ||
            this.state.user.viewerConfigs[0].type ===
              ViewerConfigTypeName.IMAGE_3D) && (
            <div>
              {this.state.session.info3D.isBoxSpan ||
              this.state.session.info3D.boxSpan !== null
                ? makeButton("Cancel", () => {
                    this.deactivateSpan()
                    alert(Severity.WARNING, "Box was not generated")
                  })
                : makeButton("Activate span", () => {
                    this.activateSpan()
                  })}
            </div>
          )}
          {this.state.task.config.tracking && (
            <div>
              {this.state.session.trackLinking
                ? makeButton("Finish", () => {
                    this.linkSelectedTracks(this.state)
                  })
                : makeButton("Link Tracks", () => {
                    this.startLinkTrack()
                  })}
            </div>
          )}
          {this.state.task.config.tracking && (
            <div>
              {makeButton("Break Track", () => {
                this.unlinkSelectedTrack(this.state)
              })}
            </div>
          )}
        </div>
        <ToastContainer hideProgressBar transition={Slide} />
      </div>
    )
  }

  /**
   * handler for the delete button/key
   *
   */
  private deletePressed(): void {
    const select = this.state.user.select
    if (Object.keys(select.labels).length > 0) {
      const item = this.state.task.items[select.item]
      const trackId = item.labels[Object.values(select.labels)[0][0]].track
      if (trackId !== undefined) {
        if (isValidId(trackId)) {
          if (!this.isKeyDown(Key.S_LOW)) {
            Session.dispatch(terminateSelectedTracks(this.state, select.item))
          } else {
            Session.dispatch(
              deleteSelectedLabelsfromTracks(this.state, select.item)
            )
          }
        } else {
          Session.dispatch(deleteSelectedLabels(this.state))
        }
      }
    }
  }

  /**
   * handles tag attribute toggle, dispatching the addLabelTag action
   *
   * @param toggleName
   * @param {string} alignment
   */
  private handleAttributeToggle(toggleName: string, alignment: string): void {
    const state = this.state
    const allAttributes = state.task.config.attributes
    const attributeIndex = this.getAttributeIndex(allAttributes, toggleName)
    if (attributeIndex === -1) {
      return
    }
    const currentAttribute = allAttributes[attributeIndex]
    const selectedIndex = currentAttribute.values.indexOf(alignment)
    if (
      state.task.config.labelTypes[state.user.select.labelType] ===
      LabelTypeName.TAG
    ) {
      Session.dispatch(addLabelTag(attributeIndex, selectedIndex))
    } else {
      const currentAttributes = state.user.select.attributes
      const attributes: { [key: number]: number[] } = {}
      for (const keyStr of Object.keys(currentAttributes)) {
        const key = Number(keyStr)
        attributes[key] = currentAttributes[key]
      }
      attributes[attributeIndex] = [selectedIndex]
      if (_.size(state.user.select.labels) > 0) {
        Session.dispatch(changeSelectedLabelsAttributes(state, attributes))
      }
      Session.dispatch(changeSelect({ attributes }))
    }
  }

  /**
   * This function updates the checked list of switch buttons.
   *
   * @param {string} switchName
   */
  private handleToggle(switchName: string): void {
    const state = this.state
    const allAttributes = state.task.config.attributes
    const toggleIndex = this.getAttributeIndex(allAttributes, switchName)
    if (toggleIndex >= 0) {
      const currentAttributes = state.user.select.attributes
      const attributes: { [key: number]: number[] } = {}
      for (const [key] of allAttributes.entries()) {
        if (key in currentAttributes) {
          attributes[key] = currentAttributes[key]
        } else {
          attributes[key] = [0]
        }
      }
      if (attributes[toggleIndex][0] > 0) {
        attributes[toggleIndex][0] = 0
      } else {
        attributes[toggleIndex][0] = 1
      }
      if (
        state.task.config.labelTypes[state.user.select.labelType] ===
        LabelTypeName.TAG
      ) {
        Session.dispatch(addLabelTag(toggleIndex, attributes[toggleIndex][0]))
      } else {
        if (_.size(state.user.select.labels) > 0) {
          Session.dispatch(changeSelectedLabelsAttributes(state, attributes))
        }
      }
      Session.dispatch(changeSelect({ attributes }))
    }
  }

  /**
   * helper function to get attribute index with respect to the label's
   * attributes
   *
   * @param name
   */
  private getAlignmentIndex(name: string): number {
    const state = this.state
    const attributeIndex = this.getAttributeIndex(
      state.task.config.attributes,
      name
    )
    if (attributeIndex < 0) {
      return 0
    }
    if (
      state.task.config.labelTypes[state.user.select.labelType] ===
      LabelTypeName.TAG
    ) {
      const item = state.task.items[state.user.select.item]
      const labelId = _.findKey(item.labels)
      if (labelId === undefined) {
        return 0
      }
      const attributes = isNaN(Number(labelId))
        ? item.labels[labelId].attributes
        : item.labels[parseInt(labelId, 10)].attributes
      const index = this.getAttributeIndex(state.task.config.attributes, name)
      if (index < 0) {
        return 0
      }
      if (attributes[index] !== undefined && attributes[index].length > 0) {
        return attributes[index][0]
      } else {
        return 0
      }
    } else {
      const currentAttributes = state.user.select.attributes
      return _.size(currentAttributes) > 0
        ? Object.keys(currentAttributes).includes(String(attributeIndex))
          ? currentAttributes[attributeIndex][0]
          : 0
        : 0
    }
  }

  /**
   * helper function to get attribute index with respect to the config
   * attributes
   *
   * @param allAttributes
   * @param name
   * @param toggleName
   */
  private getAttributeIndex(
    allAttributes: Attribute[],
    toggleName: string
  ): number {
    let attributeIndex = -1
    for (let i = 0; i < allAttributes.length; i++) {
      if (allAttributes[i].name === toggleName) {
        attributeIndex = i
      }
    }
    return attributeIndex
  }

  /**
   * Link selected tracks
   *
   * @param state
   */
  private linkSelectedTracks(state: State): void {
    const tracks = getSelectedTracks(state)

    if (tracks.length === 0) {
      alert(Severity.WARNING, "No tracks currently selected.")
    }

    if (!tracksOverlapping(tracks)) {
      Session.dispatch(mergeTracks(tracks.map((t) => t.id)))
      alert(Severity.SUCCESS, "Selected tracks have been successfuly linked.")
    } else {
      alert(Severity.WARNING, "Selected tracks have overlapping frames.")
    }
  }

  /**
   * Unlink selected track
   *
   * @param state
   */
  private unlinkSelectedTrack(state: State): void {
    const select = this.state.user.select
    const track = getSelectedTracks(state)[0]

    if (track !== undefined) {
      const newTrackId = makeTrack().id
      Session.dispatch(splitTrack(track.id, newTrackId, select.item))
    } else {
      alert(Severity.WARNING, "No tracks currently selected.")
    }
  }

  /**
   * Start to link track
   */
  private startLinkTrack(): void {
    Session.dispatch(startLinkTrack())
  }

  /**
   * Activate box spanning mode
   */
  private activateSpan(): void {
    if (!this.itemHasGroundPlane()) {
      alert(Severity.WARNING, 'First insert ground plane with "g".')
      return
    }
    Session.dispatch(activateSpan())
  }

  /**
   * Deactivate box spanning mode
   *
   * @param state
   */
  private deactivateSpan(): void {
    Session.dispatch(deactivateSpan())
  }

  /**
   * Check if current selected item has a ground plane.
   */
  private itemHasGroundPlane(): boolean {
    const selectedItem = this.state.user.select.item
    const groundPlane = Session.label3dList.getItemGroundPlane(selectedItem)
    return groundPlane !== null
  }
}
