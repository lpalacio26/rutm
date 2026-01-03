import imageUrlBuilder from "@sanity/image-url";
import { sanity } from "./sanity";

// sanity should be your configured client
const builder = imageUrlBuilder(sanity);

export function urlFor(source: any) {
  return builder.image(source);
}