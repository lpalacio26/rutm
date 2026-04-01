import {defineConfig} from 'sanity'
import {structureTool} from 'sanity/structure'
import {visionTool} from '@sanity/vision'
import {schemaTypes} from './schemaTypes'

const singletonTypes = ["homepage", "aboutPage", "tellYoursPage", "siteFooter"];

export default defineConfig({
  name: 'default',
  title: 'RUTM',

  projectId: 'ywsexjw4',
  dataset: 'production',

  plugins: [structureTool(), visionTool()],

  schema: {
    types: schemaTypes,
  },

   document: {
    // Hides the "Create new" button for singleton types
    newDocumentOptions: (prev, { creationContext }) => {
      if (creationContext.type === "global") {
        return prev.filter((item) => !singletonTypes.includes(item.templateId));
      }
      return prev;
    },
    // Removes Delete from the action menu for singletons
    actions: (prev, { schemaType }) => {
      if (singletonTypes.includes(schemaType)) {
        return prev.filter(({ action }) => action !== "delete");
      }
      return prev;
    },
  },
});