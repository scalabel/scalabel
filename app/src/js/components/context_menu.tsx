import { Grid } from '@material-ui/core'
import ExpansionPanel from '@material-ui/core/ExpansionPanel'
import ExpansionPanelDetails from '@material-ui/core/ExpansionPanelDetails'
import ExpansionPanelSummary from '@material-ui/core/ExpansionPanelSummary'
import TextField from '@material-ui/core/TextField'
import ExpandMoreIcon from '@material-ui/icons/ExpandMore'
import { Autocomplete } from '@material-ui/lab'
import * as React from 'react'
import { changeSelect } from '../action/common'
import { changeSelectedLabelsCategories } from '../action/select'
import Session from '../common/session'
import { LabelTypeName } from '../common/types'
import Label3D from '../drawable/3d/label3d'
import { Box3dMenu } from './ box3d_menu'
import { AttributeSelector } from './attribute_selector'

/** Make label menu */
function makeLabelMenu (label: Label3D) {
  switch (label.type) {
    case LabelTypeName.BOX_3D:
      return <Box3dMenu label={label} />
  }
  return null
}

/**
 * Function to make collapsible section with title,
 * for use with React state hook
 */

/** Context menu class */
export class ContextMenu extends React.Component {
  constructor () {
    super({})
  }

  /** Render context menu */
  public render () {
    const categories = Session.getState().task.config.categories
    const selectedLabel = Session.label3dList.selectedLabel
    let labelMenu = null
    if (selectedLabel) {
      labelMenu = makeLabelMenu(selectedLabel)
    }
    return (
      <Grid
        justify={'flex-start'}
        container
        direction='row'
      >
        <Autocomplete
          options={categories}
          getOptionLabel={(x) => x}
          id='contextMenuCategories'
          style={{ width: '100%' }}
          renderInput={
            (params) =>
              <TextField
                {...params}
                margin='dense'
                label='Category'
                fullWidth
              />
          }
          disabled={!selectedLabel}
          onChange={(_e: React.ChangeEvent<{}>, val: string | null) => {
            if (val) {
              const state = Session.getState()
              const categoryId =
                state.task.config.categories.indexOf(val)
              Session.dispatch(changeSelect({ category: categoryId }))
              // update categories if any labels are selected
              if (Object.keys(state.user.select.labels).length > 0) {
                Session.dispatch(changeSelectedLabelsCategories(
                  state, [categoryId]
                ))
              }
            }
          }}
          value={(selectedLabel) ? categories[selectedLabel.category[0]] : null}
        />
        <ExpansionPanel>
          <ExpansionPanelSummary
            expandIcon={<ExpandMoreIcon />}
          >
            Attributes
          </ExpansionPanelSummary>
          <ExpansionPanelDetails>
            <AttributeSelector/>
          </ExpansionPanelDetails>
        </ExpansionPanel>
        {labelMenu}
      </Grid>
    )
  }
}
