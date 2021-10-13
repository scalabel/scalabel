import ListItem from "@material-ui/core/ListItem/ListItem"
import React from "react"

import { ToolbarCategory } from "../components/toolbar_category"
import { ListButton } from "../components/toolbar_list_button"
import { SwitchButton } from "../components/toolbar_switch"
import { AttributeToolType } from "../const/common"

/**
 * This is renderTemplate function that renders the category.
 *
 * @param {string} type
 * @param {Function} handleToggle
 * @param handleAttributeToggle
 * @param getAlignmentIndex
 * @param {string} name
 * @param {string[]} options
 */
export function renderTemplate(
  type: string,
  handleToggle: (switchName: string) => void,
  handleAttributeToggle: (toggleName: string, alignment: string) => void,
  getAlignmentIndex: (switName: string) => number,
  name: string,
  options: string[]
): React.ReactNode {
  if (type === AttributeToolType.SWITCH) {
    return (
      <SwitchButton
        onChange={handleToggle}
        name={name}
        getAlignmentIndex={getAlignmentIndex}
      />
    )
  } else if (type === AttributeToolType.LIST) {
    return (
      <ListItem dense={true} style={{ textAlign: "center" }}>
        <ListButton
          name={name}
          values={options}
          handleAttributeToggle={handleAttributeToggle}
          getAlignmentIndex={getAlignmentIndex}
        />
      </ListItem>
    )
  } else if (type === AttributeToolType.LONG_LIST) {
    return (
      <ListItem dense={true} style={{ textAlign: "center" }}>
        <ToolbarCategory
          categories={options}
          treeCategories={null}
          headerText={name}
        />
      </ListItem>
    )
  }
}
