import { defineType, defineField } from "sanity";

export default defineType({
  name: 'imageBlock',
  title: 'Image',
  type: 'object',
  fields: [
    defineField({ name: 'image', title: 'Image', type: 'image', options: { hotspot: true } }),
    defineField({ name: 'caption', title: 'Caption', type: 'string' }),
    defineField({
      name: 'size',
      title: 'Size',
      type: 'string',
      options: {
        list: [
          { title: 'Normal (content width)', value: 'normal' },
          { title: 'Wide (bleed past text column)', value: 'wide' },
        ],
        layout: 'radio',
      },
      initialValue: 'normal',
    }),
  ],
})