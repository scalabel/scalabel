import List from "@material-ui/core/List/List"
// import ListItem from "@material-ui/core/ListItem"
import _ from "lodash"
import React from "react"

import {
  changeSelect,
  changeViewerConfig,
  mergeTracks,
  startLinkTrack
} from "../action/common"
import {
  changeSelectedLabelsAttributes,
  deleteSelectedLabels,
  terminateSelectedTracks
} from "../action/select"
import { addLabelTag } from "../action/tag"
import { renderTemplate } from "../common/label"
import Session from "../common/session"
import { Key, LabelTypeName } from "../const/common"
import { getSelectedTracks } from "../functional/state_util"
import { isValidId } from "../functional/states"
import { tracksOverlapping } from "../functional/track"
import { Attribute, State } from "../types/state"
import { makeButton } from "./button"
import { Component } from "./component"
import { Category } from "./toolbar_category"

/** This is the interface of props passed to ToolBar */
interface Props {
  /** categories of ToolBar */
  categories: string[] | null
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
    const { categories, attributes } = this.props
    return (
      <div>
        {categories !== null ? (
          <Category categories={categories} headerText={"Category"} />
        ) : null}
        <List>
          {attributes.map((element: Attribute) => (
            <React.Fragment key={element.name}>
              {renderTemplate(
                element.toolType,
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
        </div>
      </div>
    )
  }

  /**
   * handler for the delete button/key
   *
   * @param {string} alignment
   */
  private deletePressed(): void {
    const select = this.state.user.select
    if (Object.keys(select.labels).length > 0) {
      const item = this.state.task.items[select.item]
      if (isValidId(item.labels[Object.values(select.labels)[0][0]].track)) {
        Session.dispatch(terminateSelectedTracks(this.state, select.item))
      } else {
        Session.dispatch(deleteSelectedLabels(this.state))
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

    if (!tracksOverlapping(tracks)) {
      Session.dispatch(mergeTracks(tracks.map((t) => t.id)))
    }
  }

  /**
   * Start to link track
   */
  private startLinkTrack(): void {
    Session.dispatch(startLinkTrack())
  }
}
