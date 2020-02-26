import List from '@material-ui/core/List/List'
import _ from 'lodash'
import React from 'react'
import { changeSelect } from '../action/common'
import { changeSelectedLabelsAttributes } from '../action/select'
import { addLabelTag } from '../action/tag'
import { renderTemplate } from '../common/label'
import Session from '../common/session'
import { LabelTypeName } from '../common/types'
import { Attribute } from '../functional/types'
import { Component } from './component'

/** Attribute selection menu */
export class AttributeSelector extends Component<{}> {
  constructor () {
    super({})
    this.handleAttributeToggle = this.handleAttributeToggle.bind(this)
    this.getAlignmentIndex = this.getAlignmentIndex.bind(this)
    this.handleToggle = this.handleToggle.bind(this)
  }

  /** Render function */
  public render () {
    const attributes = this.state.task.config.attributes
    return (
      <List>
        {attributes.map((element) =>
          (<React.Fragment key = {element.name}>
          {renderTemplate(
            element.toolType,
            this.handleToggle,
            this.handleAttributeToggle,
            this.getAlignmentIndex,
            element.name,
            element.values
          )}
          </React.Fragment>
          )
        )}
      </List>
    )
  }

  /**
   * handles tag attribute toggle, dispatching the addLabelTag action
   * @param {string} alignment
   */
  private handleAttributeToggle (toggleName: string, alignment: string) {
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
   * @param {string} switchName
   */
  private handleToggle (switchName: string) {
    const state = this.state
    const allAttributes = state.task.config.attributes
    const toggleIndex = this.getAttributeIndex(allAttributes, switchName)
    if (toggleIndex >= 0) {
      const currentAttributes = state.user.select.attributes
      const attributes: { [key: number]: number[] } = {}
      for (const [key] of allAttributes.entries()) {
        if (currentAttributes[key]) {
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
   */
  private getAlignmentIndex (name: string): number {
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
      const labelId = Number(_.findKey(item.labels))
      if (isNaN(labelId)) {
        return 0
      }
      const attributes = item.labels[labelId].attributes
      const index = this.getAttributeIndex(state.task.config.attributes, name)
      if (index < 0) {
        return 0
      }
      if (attributes[index]) {
        return attributes[index][0]
      } else {
        return 0
      }
    } else {
      const currentAttributes = state.user.select.attributes
      return currentAttributes
        ? Object.keys(currentAttributes).indexOf(String(attributeIndex)) >= 0
          ? currentAttributes[attributeIndex][0]
          : 0
        : 0
    }
  }
  /**
   * helper function to get attribute index with respect to the config
   * attributes
   * @param allAttributes
   * @param name
   */
  private getAttributeIndex (allAttributes: Attribute[], toggleName: string
  ): number {
    let attributeIndex = -1
    for (let i = 0; i < allAttributes.length; i++) {
      if (allAttributes[i].name === toggleName) {
        attributeIndex = i
      }
    }
    return attributeIndex
  }
}
