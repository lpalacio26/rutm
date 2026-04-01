import { defineType, defineField } from "sanity"

export default defineType({
  name: "tellYoursPage",
  title: "Tell Yours Page",
  type: "document",
  fields: [

  
    defineField({
      name: "body",
      title: "Body text",
      type: "array",
      of: [{ type: "block" }],
    }),
    defineField({
      name: "marqueeText",
      title: "Marquee banner text",
      type: "string",
      initialValue: "TELL YOURS",
    }),
      defineField({
      name: "marqueeSpeed",
      title: "Marquee speed (px/s)",
      type: "number",
      initialValue: 40,
    }),
  ],
})