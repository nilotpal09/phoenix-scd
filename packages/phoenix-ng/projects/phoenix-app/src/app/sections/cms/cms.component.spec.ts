import { ComponentFixture, TestBed, waitForAsync } from '@angular/core/testing';

import { CMSComponent } from './cms.component';
import { AppModule } from '../../app.module';
import { EventDisplayService } from 'phoenix-ui-components';
import { HttpClient } from '@angular/common/http';
import { ScriptLoader } from 'phoenix-event-display';

describe('CMSComponent', () => {
  let component: CMSComponent;
  let fixture: ComponentFixture<CMSComponent>;

  const mock = jasmine.createSpyObj('', ['NewHttpRequest']);
  mock.NewHttpRequest.and.callFake(() =>
    jasmine.createSpyObj('returnValue', ['send'])
  );

  let eventDisplayService: EventDisplayService;

  beforeAll(() => {
    spyOn(ScriptLoader, 'loadScripts').and.resolveTo(mock);
  });

  beforeEach(waitForAsync(() => {
    TestBed.configureTestingModule({
      imports: [AppModule],
      providers: [HttpClient, EventDisplayService],
    }).compileComponents();

    eventDisplayService = TestBed.get(EventDisplayService);
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(CMSComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  // Test if three.js is initialized
  it('should initialize three.js canvas', () => {
    spyOn(eventDisplayService, 'parsePhoenixEvents').and.stub();
    component.ngOnInit();
    expect(document.getElementById('three-canvas')).toBeTruthy();
  });
});
