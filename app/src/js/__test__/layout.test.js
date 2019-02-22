import React from 'react';
import {mount, configure} from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';
import LabelLayout from '../components/label_layout';
import TitleBar from '../components/title_bar';

configure({adapter: new Adapter()});
let titleBar = (
    <TitleBar
        title='Test Title'
        instructionLink='https://www.scalabel.ai'
        dashboardLink={'/vendor?project_name=test'}
    />
);

const leftSidebar1 = (<div>0</div>);
const leftSidebar2 = (<div>1</div>);
const main = (<div>2</div>);
const bottomBar = (<div>3</div>);
const rightSidebar1 = (<div>4</div>);
const rightSidebar2 = (<div>5</div>);

// L: left sidebar; B: bottom bar; R: right sidebar
describe('Layout Tests', function() {
  // Two blocks in left sidebar, bottom bar, and two blocks in right sidebar
  it('L2BR2', function() {
    let layout = (<LabelLayout
        titleBar={titleBar}
        leftSidebar1={leftSidebar1}
        leftSidebar2={leftSidebar2}
        bottomBar={bottomBar}
        main={main}
        rightSidebar1={rightSidebar1}
        rightSidebar2={rightSidebar2}
    />);
    const component = mount(layout);
    expect(component.find('.titleBar').length).toBe(1);
    expect(component.find('.leftSidebar1').length).toBe(1);
    expect(component.find('.leftSidebar2').length).toBe(1);
    expect(component.find('.bottomBar').length).toBe(1);
    expect(component.find('.main').length).toBe(1);
    expect(component.find('.rightSidebar1').length).toBe(1);
    expect(component.find('.rightSidebar2').length).toBe(1);
  });

  // one block in left sidebar
  it('L1', function() {
    let layout = (<LabelLayout
        titleBar={titleBar}
        leftSidebar1={leftSidebar1}
        main={main}
    />);
    const component = mount(layout);
    expect(component.find('.titleBar').length).toBe(1);
    expect(component.find('.leftSidebar1').length).toBe(1);
    expect(component.find('.leftSidebar2').length).toBe(0);
    expect(component.find('.bottomBar').length).toBe(0);
    expect(component.find('.main').length).toBe(1);
    expect(component.find('.rightSidebar1').length).toBe(0);
    expect(component.find('.rightSidebar2').length).toBe(0);
  });

  // two blocks in left sidebar and one block in right sidebar
  it('L2R1', function() {
    let layout = (<LabelLayout
        titleBar={titleBar}
        leftSidebar1={leftSidebar1}
        leftSidebar2={leftSidebar2}
        main={main}
        rightSidebar1={rightSidebar1}
    />);
    const component = mount(layout);
    expect(component.find('.titleBar').length).toBe(1);
    expect(component.find('.leftSidebar1').length).toBe(1);
    expect(component.find('.leftSidebar2').length).toBe(1);
    expect(component.find('.bottomBar').length).toBe(0);
    expect(component.find('.main').length).toBe(1);
    expect(component.find('.rightSidebar1').length).toBe(1);
    expect(component.find('.rightSidebar2').length).toBe(0);
  });

  // two blocks in left sidebar and bottom bar
  it('L2B', function() {
    let layout = (<LabelLayout
        titleBar={titleBar}
        leftSidebar1={leftSidebar1}
        leftSidebar2={leftSidebar2}
        bottomBar={bottomBar}
        main={main}
    />);
    const component = mount(layout);
    expect(component.find('.titleBar').length).toBe(1);
    expect(component.find('.leftSidebar1').length).toBe(1);
    expect(component.find('.leftSidebar2').length).toBe(1);
    expect(component.find('.bottomBar').length).toBe(1);
    expect(component.find('.main').length).toBe(1);
    expect(component.find('.rightSidebar1').length).toBe(0);
    expect(component.find('.rightSidebar2').length).toBe(0);
  });

  // one block in left sidebar, bottom bar, and two blocks in right sidebar
  it('L1BR2', function() {
    let layout = (<LabelLayout
        titleBar={titleBar}
        leftSidebar1={leftSidebar1}
        bottomBar={bottomBar}
        main={main}
        rightSidebar1={rightSidebar1}
        rightSidebar2={rightSidebar2}
    />);
    const component = mount(layout);
    expect(component.find('.titleBar').length).toBe(1);
    expect(component.find('.leftSidebar1').length).toBe(1);
    expect(component.find('.leftSidebar2').length).toBe(0);
    expect(component.find('.bottomBar').length).toBe(1);
    expect(component.find('.main').length).toBe(1);
    expect(component.find('.rightSidebar1').length).toBe(1);
    expect(component.find('.rightSidebar2').length).toBe(1);
  });
});
