import * as fa from "@fortawesome/free-solid-svg-icons"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import {
  Button,
  Grid,
  IconButton,
  List,
  Table,
  TableCell,
  TableHead,
  TableRow
} from "@material-ui/core"
import Chip from "@material-ui/core/Chip"
import ListItemText from "@material-ui/core/ListItemText"
import TableBody from "@material-ui/core/TableBody"
import Typography from "@material-ui/core/Typography"
import CloudDownloadIcon from "@material-ui/icons/CloudDownload"
import { withStyles } from "@material-ui/styles"
import React from "react"

import { QueryArg } from "../const/common"
import { Endpoint } from "../const/connection"
import {
  dashboardWindowStyles,
  headerStyle,
  listEntryStyle,
  sidebarStyle
} from "../styles/dashboard"
import { SubmitData } from "../types/state"
import DividedPage from "./divided_page"
import { formatDate, getSubmissionTime } from "./util"

export interface ProjectOptions {
  /** project name */
  name: string
  /** item type */
  itemType: string
  /** label type */
  labelTypes: string[]
  /** task size */
  taskSize: number
  /** number of items */
  numItems: number
  /** number of categories */
  numLeafCategories: number
  /** number of attributes */
  numAttributes: number
}

export interface TaskOptions {
  /** number of labeled images */
  numLabeledItems: string
  /** number of labels */
  numLabels: string
  /** list of timestamped submissions */
  submissions: SubmitData[]
  /** task link handler url */
  handlerUrl: string
}

export interface DashboardContents {
  /** project metadata */
  projectMetaData: ProjectOptions
  /** tasks */
  taskMetaDatas: TaskOptions[]
  /** num users */
  numUsers: number
}

interface DashboardClassType {
  /** root */
  root: string
  /** table row */
  row: string
  /** task link button */
  linkButton: string
  /** table header cell */
  headerCell: string
  /** table body cell */
  bodyCell: string
}

interface DashboardProps {
  /** Create classes */
  classes: DashboardClassType
  /** dashboard contents */
  dashboardContents: DashboardContents
  /** if this is the vendor dashboard */
  vendor?: boolean
}

interface HeaderProps {
  /** header classes */
  classes: HeaderClassType
  /** total tasks */
  totalTaskLabeled: number
  /** total labels */
  totalLabels: number
  /** number of users currently editing */
  numUsers: number
  /** if this is the vendor dashboard */
  vendor?: boolean
}

interface HeaderClassType {
  /** flex grow buffer style */
  grow: string
  /** class type for chip */
  chip: string
}

interface SidebarProps {
  /** sidebar classes */
  classes: SidebarClassType
  /** project metadata */
  projectMetaData: ProjectOptions
  /** if this is the vendor dashboard */
  vendor?: boolean
}

interface SidebarClassType {
  root: string
  /** list root */
  listRoot: string
  /** list item */
  listItem: string
  /** colored list item */
  coloredListItem: string
  /** link class */
  link: string
}

interface ListEntryProps {
  /** sidebar classes */
  classes: ListEntryClassType
  /** entry tag */
  tag: string
  /** entry value */
  entry: string | number
}

interface ListEntryClassType {
  /** list tag */
  listTag: string
  /** list entry */
  listEntry: string
  /** list grid container */
  listContainer: string
}

/**
 * creates the dashboard component
 *
 * @param props
 * @constructor
 */
function Dashboard(props: DashboardProps): JSX.Element {
  const { classes, vendor } = props
  let totalTaskLabeled = 0
  let totalLabels = 0
  const projectMetaData = props.dashboardContents.projectMetaData
  const taskMetaDatas = props.dashboardContents.taskMetaDatas
  const numUsers = props.dashboardContents.numUsers
  const sidebarContent = (
    <StyledSidebar projectMetaData={projectMetaData} vendor={vendor} />
  )
  const align = "center"
  const mainContent = (
    <div className={classes.root}>
      <Table size="small" stickyHeader={true}>
        <TableHead>
          <TableRow>
            <TableCell align={align} className={classes.headerCell}>
              {"Task Index"}
            </TableCell>
            <TableCell align={align} className={classes.headerCell}>
              {"# Labeled Images"}
            </TableCell>
            <TableCell align={align} className={classes.headerCell}>
              {"# Labels"}
            </TableCell>
            <TableCell align={align} className={classes.headerCell}>
              {"Submitted"}
            </TableCell>
            <TableCell align={align} className={classes.headerCell}>
              {"Task Link"}
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {taskMetaDatas.map((value: TaskOptions, index) => {
            const time = getSubmissionTime(value.submissions)
            let dateString = ""
            if (time !== -1) {
              dateString = formatDate(time)
            }
            totalLabels += Number(value.numLabels)
            totalTaskLabeled += Number(value.numLabeledItems) > 0 ? 1 : 0
            return (
              <TableRow
                key={index}
                className={index % 2 === 0 ? classes.row : ""}
              >
                <TableCell className={classes.bodyCell} align={align}>
                  {index}
                </TableCell>
                <TableCell
                  className={classes.bodyCell}
                  align={align}
                  data-testid={"num-labeled-images-" + index.toString()}
                >
                  {value.numLabeledItems}
                </TableCell>
                <TableCell
                  className={classes.bodyCell}
                  align={align}
                  data-testid={"num-labels-" + index.toString()}
                >
                  {value.numLabels}
                </TableCell>
                <TableCell
                  className={classes.bodyCell}
                  align={align}
                  data-testid={"submitted-" + index.toString()}
                >
                  {dateString}
                </TableCell>
                <TableCell className={classes.bodyCell} align={align}>
                  <IconButton
                    className={classes.linkButton}
                    color="inherit"
                    href={
                      `./${value.handlerUrl}` +
                      `?${QueryArg.PROJECT_NAME}=${projectMetaData.name}` +
                      `&${QueryArg.TASK_INDEX}=${index}`
                    }
                    data-testid={"task-link-" + index.toString()}
                  >
                    <FontAwesomeIcon
                      icon={fa.faExternalLinkAlt}
                      size="1x"
                      transform="grow-6"
                    />
                  </IconButton>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </div>
  )
  const headerContent = (
    <StyledHeader
      totalLabels={totalLabels}
      totalTaskLabeled={totalTaskLabeled}
      numUsers={numUsers}
      vendor={vendor}
    />
  )
  /**
   * renders the dashboard
   *
   * @return component
   */
  return (
    <DividedPage
      header={headerContent}
      sidebar={sidebarContent}
      main={mainContent}
    />
  )
}

/**
 * creates the header
 *
 * @param props
 */
function header(props: HeaderProps): JSX.Element {
  const { classes, totalLabels, totalTaskLabeled, numUsers, vendor } = props
  return (
    <>
      <Typography variant="h6" noWrap>
        {vendor !== undefined && vendor
          ? "Vendor Dashboard"
          : "Project Dashboard"}
      </Typography>
      <div className={classes.grow} />
      {vendor !== undefined && vendor ? null : (
        <React.Fragment>
          <Typography variant="body1" noWrap>
            Labeled Tasks
          </Typography>
          <Chip
            label={totalTaskLabeled}
            className={classes.chip}
            data-testid="total-tasks"
          />
          <Typography variant="body1" noWrap>
            Total Labels
          </Typography>
          <Chip
            label={totalLabels}
            className={classes.chip}
            data-testid="total-labels"
          />
          <Typography variant="body1" noWrap>
            Users
          </Typography>
          <Chip
            label={numUsers}
            className={classes.chip}
            data-testid="num-users"
          />
        </React.Fragment>
      )}
    </>
  )
}

/**
 * creates the sidebar
 *
 * @param props
 */
function sidebar(props: SidebarProps): JSX.Element {
  const { classes, projectMetaData, vendor } = props
  const sidebarListItems = [
    { tag: "Project Name", entry: projectMetaData.name },
    { tag: "Item Type", entry: projectMetaData.itemType },
    { tag: "Label Type", entry: projectMetaData.labelTypes[0] },
    { tag: "Task Size", entry: projectMetaData.taskSize },
    { tag: "# Items", entry: projectMetaData.numItems },
    { tag: "# Categories", entry: projectMetaData.numLeafCategories },
    { tag: "# Attributes", entry: projectMetaData.numAttributes }
  ]
  return (
    <>
      <List className={classes.listRoot}>
        {sidebarListItems.map((value, index) => (
          <ListItemText
            key={value.tag}
            className={
              index % 2 === 0
                ? `${classes.listItem} ${classes.coloredListItem}`
                : classes.listItem
            }
            primary={<StyledListEntry tag={value.tag} entry={value.entry} />}
          />
        ))}
      </List>
      {vendor !== undefined && vendor ? null : (
        <>
          <Button
            variant="contained"
            color="primary"
            className={classes.link}
            startIcon={<CloudDownloadIcon />}
            href={`.${Endpoint.EXPORT}?project_name=` + projectMetaData.name}
          >
            DOWNLOAD LABELS
          </Button>
          {/* <Link
            variant="body2"
            className={classes.link}
            color="inherit"
            href={"./postDownloadTaskURL?project_name=" + projectMetaData.name}
            data-testid="download-link"
          >
            DOWNLOAD ASSIGNMENT URLS
          </Link> */}
        </>
      )}
    </>
  )
}

/**
 * sidebar list entry
 *
 * @param props
 */
function listEntry(props: ListEntryProps): JSX.Element {
  const { classes, tag, entry } = props
  return (
    <React.Fragment>
      <Grid
        spacing={1}
        alignItems={"baseline"}
        justify={"space-around"}
        className={classes.listContainer}
        container
      >
        <Grid item xs>
          <Typography className={classes.listTag} variant="body2">
            {tag}
          </Typography>
        </Grid>
        <Grid item xs>
          <Typography className={classes.listEntry} variant="body2">
            {entry}
          </Typography>
        </Grid>
      </Grid>
    </React.Fragment>
  )
}

/** export sub-components for testing */
export const StyledHeader = withStyles(headerStyle)(header)
export const StyledSidebar = withStyles(sidebarStyle)(sidebar)
const StyledListEntry = withStyles(listEntryStyle)(listEntry)
/** export dashboard page */
export default withStyles(dashboardWindowStyles)(Dashboard)
