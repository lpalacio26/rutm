import { defineType, defineField } from "sanity"

export default defineType({
  name: "aboutPage",
  title: "About Page",
  type: "document",
  fields: [
    defineField({
      name: "definitionBlocks",
      title: "Definition blockquotes",
      description: "The quoted definitions — etymology first, then Cambridge entries",
      type: "array",
      of: [{ type: "text" }],
    }),
    defineField({
      name: "dictionarySource",
      title: "Dictionary source attribution",
      type: "string",
      initialValue: "Cambridge Advanced Learner's Dictionary & Thesaurus © Cambridge University Press",
    }),
    defineField({
      name: "body",
      title: "Body text",
      type: "array",
      of: [{ type: "block" }],
    }),
  ],
})